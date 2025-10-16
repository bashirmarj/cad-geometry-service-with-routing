"""
Vectis Manufacturing - CAD Geometry Analysis Microservice (Render 512MB Safe Mode)
-------------------------------------------------------------------------------
- Parses STEP files using pythonOCC
- Extracts volume, area, faces, and complexity
- Selects machining routing and estimates cost
- Tessellates geometry safely within Render memory limits
- Adds robust BRep feature-edge extraction and simplification
- ✅ Now returns feature_edges in the JSON response for the viewer
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile, os, logging, math, numpy as np

# --- pythonOCC imports ---
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import (
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Line,
    GeomAbs_Circle,
    GeomAbs_Ellipse,
    GeomAbs_BSplineCurve,
    GeomAbs_BezierCurve,
)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt

# -------------------------------------------------------------------------
# Flask setup
# -------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# INLINE ROUTING + COST LOGIC
# -------------------------------------------------------------------------
def select_routings_industrial(geometry_descriptor, material):
    is_cyl = geometry_descriptor.get("is_cylindrical", False)
    comp = geometry_descriptor.get("complexity_score", 5)
    holes = geometry_descriptor.get("holes_count", 0)
    grooves = geometry_descriptor.get("grooves_count", 0)

    if is_cyl and holes <= 2:
        routes, reason = ["Lathe"], "Cylindrical → Lathe"
    elif is_cyl and grooves > 0:
        routes, reason = ["Mill-Turn"], "Cylindrical with grooves → Mill-Turn"
    elif not is_cyl and comp <= 5:
        routes, reason = ["VMC 3-axis"], "Prismatic, moderate → VMC 3-axis"
    else:
        routes, reason = ["HMC 4-axis"], "Complex prismatic → HMC 4-axis"

    logger.info(f"Routing: {routes} | {reason}")
    return {"recommended_routings": routes, "reasoning": reason}


def estimate_machining_time_and_cost(geometry_descriptor, material, routings):
    base_rate = 60.0  # USD/hr
    vol = geometry_descriptor.get("volume_cm3", 0)
    comp = geometry_descriptor.get("complexity_score", 5)
    route = routings[0] if routings else "VMC 3-axis"

    t_hr = (vol / 500.0) * (comp / 5.0)
    factor = {"Lathe": 0.8, "Mill-Turn": 1.1, "HMC 4-axis": 1.3, "VMC 3-axis": 1.0}.get(route, 1.0)
    t_hr *= factor
    cost = t_hr * base_rate

    summary = f"{t_hr:.2f} hr using {route} on {material}"
    logger.info(f"Machining: {summary}, ${cost:.2f}")
    return {"machining_summary": summary, "total_cost_usd": round(cost, 2)}

# -------------------------------------------------------------------------
# --- Feature Edge Extraction + Simplification ---
# -------------------------------------------------------------------------
def extract_feature_edges(shape, n_points=20, angle_tol_deg=3.0, merge_dist=0.05):
    edges = []
    stats = {"Line": 0, "Circle": 0, "Ellipse": 0, "BSpline": 0, "Bezier": 0, "Other": 0}
    try:
        exp = TopExp_Explorer(shape, TopAbs_EDGE)
        while exp.More():
            edge = exp.Current()
            try:
                curve_data = BRep_Tool.Curve(edge)
                if len(curve_data) == 3:
                    curve_handle, first, last = curve_data
                elif len(curve_data) == 2:
                    curve_handle, (first, last) = curve_data
                else:
                    exp.Next()
                    continue
            except Exception:
                exp.Next()
                continue

            if not curve_handle:
                exp.Next()
                continue

            curve_adapter = BRepAdaptor_Curve(edge)
            curve_type = curve_adapter.GetType()
            if curve_type == GeomAbs_Line:
                stats["Line"] += 1
                num_samples = 2
            elif curve_type == GeomAbs_Circle:
                stats["Circle"] += 1
                num_samples = 16
            elif curve_type == GeomAbs_Ellipse:
                stats["Ellipse"] += 1
                num_samples = 16
            elif curve_type == GeomAbs_BSplineCurve:
                stats["BSpline"] += 1
                num_samples = 12
            elif curve_type == GeomAbs_BezierCurve:
                stats["Bezier"] += 1
                num_samples = 12
            else:
                stats["Other"] += 1
                num_samples = 10

            pts = []
            for t in np.linspace(first, last, num_samples):
                try:
                    p = gp_Pnt()
                    curve_handle.D0(t, p)
                    pts.append([p.X(), p.Y(), p.Z()])
                except Exception:
                    continue

            if len(pts) >= 2:
                simplified = simplify_polyline(pts, angle_tol_deg, merge_dist)
                edges.append(simplified)
            exp.Next()

        logger.info(
            f"Extracted {len(edges)} feature edges "
            f"(Lines={stats['Line']}, Circles={stats['Circle']}, "
            f"Splines={stats['BSpline']}, Ellipses={stats['Ellipse']}, "
            f"Bezier={stats['Bezier']}, Other={stats['Other']})"
        )
    except Exception as e:
        logger.warning(f"Feature edge extraction failed: {e}")
    return edges


def simplify_polyline(points, angle_tol_deg=3.0, merge_dist=0.05):
    if len(points) <= 2:
        return points
    simplified = [points[0]]
    for i in range(1, len(points) - 1):
        p_prev = np.array(simplified[-1])
        p_curr = np.array(points[i])
        p_next = np.array(points[i + 1])
        v1, v2 = p_curr - p_prev, p_next - p_curr
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 < merge_dist or norm2 < merge_dist:
            continue
        dot = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        ang = math.degrees(math.acos(dot))
        if ang > angle_tol_deg:
            simplified.append(points[i])
    simplified.append(points[-1])
    return simplified

# -------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route("/analyze-cad", methods=["POST"])
def analyze_cad():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size > 25_000_000:
        return jsonify({"error": "File too large for Render plan"}), 400

    material = request.form.get("material", "Cold Rolled Steel")
    tolerance = float(request.form.get("tolerance", 0.02))
    quality = float(request.form.get("quality", 0.999))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".stp") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        reader = STEPControl_Reader()
        if reader.ReadFile(tmp_path) != 1:
            raise Exception("Invalid STEP file")
        reader.TransferRoots()
        shape = reader.OneShape()
        if shape.IsNull():
            raise Exception("No shape found")

        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)
        volume_mm3 = props.Mass()
        brepgprop.SurfaceProperties(shape, props)
        area_mm2 = props.Mass()

        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        width, height, depth = xmax - xmin, ymax - ymin, zmax - zmin

        cyl_faces = plan_faces = total_faces = 0
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            surf = BRepAdaptor_Surface(exp.Current())
            t = surf.GetType()
            if t == GeomAbs_Cylinder:
                cyl_faces += 1
            elif t == GeomAbs_Plane:
                plan_faces += 1
            total_faces += 1
            exp.Next()
        is_cyl = cyl_faces > total_faces * 0.4
        comp = min(10, max(1, int(total_faces / 10) + 3))

        geom = {
            "volume_cm3": round(volume_mm3 / 1000, 2),
            "bounding_box": [width, height, depth],
            "is_cylindrical": is_cyl,
            "holes_count": 0,
            "grooves_count": 0,
            "complexity_score": comp,
            "tolerance": tolerance,
        }

        mesh = tessellate_shape(shape, quality, geom)
        feature_edges = extract_feature_edges(shape)

        routing = select_routings_industrial(geom, material)
        estimate = estimate_machining_time_and_cost(geom, material, routing["recommended_routings"])

        # ✅ Include feature_edges directly in the returned JSON
        result = {
            "volume_cm3": geom["volume_cm3"],
            "surface_area_cm2": round(area_mm2 / 100, 2),
            "part_width_cm": round(width / 10, 2),
            "part_height_cm": round(height / 10, 2),
            "part_depth_cm": round(depth / 10, 2),
            "complexity_score": comp,
            "is_cylindrical": is_cyl,
            "planar_faces": plan_faces,
            "cylindrical_faces": cyl_faces,
            "total_faces": total_faces,
            "mesh_data": mesh,
            "feature_edges": feature_edges,   # ✅ Added here
            "recommended_routings": routing["recommended_routings"],
            "routing_reasoning": routing["reasoning"],
            "machining_summary": estimate["machining_summary"],
            "estimated_total_cost_usd": estimate["total_cost_usd"],
        }

        logger.info(f"Returning {len(feature_edges)} feature edges to frontend")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error analyzing file: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

# -------------------------------------------------------------------------
def tessellate_shape(shape, quality=0.999, geometry_descriptor=None):
    try:
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        size_mm = max(xmax - xmin, ymax - ymin, zmax - zmin)

        min_defl = 0.15
        max_defl = 1.0
        base_defl = 0.4 * (10 ** (-(quality * 2.5)))
        base_defl = min(max(base_defl, min_defl), max_defl)

        comp = geometry_descriptor.get("complexity_score", 5) if geometry_descriptor else 5
        comp_factor = 1.0 + (comp / 20.0)
        defl = min(base_defl * comp_factor, max_defl)
        ang_defl = 0.08

        logger.info(f"[SAFE MODE] defl={defl:.3f}mm, ang={ang_defl:.3f}, size={size_mm:.1f}mm, comp={comp}")

        mesh = BRepMesh_IncrementalMesh(shape, defl, False, ang_defl, True)
        mesh.Perform()
        if not mesh.IsDone():
            raise Exception("Mesh failed")

        verts, inds, vmap, vidx = [], [], {}, 0
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = exp.Current()
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation(face, loc)
            if tri is None:
                exp.Next()
                continue
            tr = loc.Transformation()
            rev = face.Orientation() == 1
            fverts = []
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i)
                p.Transform(tr)
                key = (round(p.X(), 6), round(p.Y(), 6), round(p.Z(), 6))
                if key not in vmap:
                    verts.extend([p.X(), p.Y(), p.Z()])
                    vmap[key] = vidx
                    vidx += 1
                fverts.append(vmap[key])
            for i in range(1, tri.NbTriangles() + 1):
                t = tri.Triangle(i)
                n1, n2, n3 = t.Get()
                a, b, c = fverts[n1 - 1], fverts[n2 - 1], fverts[n3 - 1]
                inds.extend([a, c, b] if rev else [a, b, c])
            exp.Next()

        logger.info(f"Mesh complete: {len(verts)//3} verts, {len(inds)//3} tris")
        return {"vertices": verts, "indices": inds, "normals": []}
    except Exception as e:
        logger.error(f"Tessellation error: {e}")
        return {"vertices": [], "indices": [], "normals": []}

# -------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
