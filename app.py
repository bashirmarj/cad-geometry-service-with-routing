"""
Unified CAD Geometry Analysis and Industrial Machining Quotation System
Parses STEP/IGES files, detects geometry, selects machining routings, estimates cost, and generates mesh data.
Ready for deployment on Render.com
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile, os, logging, math

# OCC imports for geometry parsing
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# ROUTING SELECTION MODULE
# =============================

def select_routings_industrial(desc, material):
    """Select manufacturing routings based on geometry, material, and tolerance"""
    routings, reasons = [], []
    bbox = desc.get("bounding_box", [0, 0, 0])
    largest_dim = max(bbox) if bbox else 0
    volume = desc.get("volume_cm3", 0)
    complexity = desc.get("complexity_score", 5)
    holes = desc.get("holes_count", 0)
    grooves = desc.get("grooves_count", 0)
    tolerance = desc.get("tolerance", 0.05)
    is_cylindrical = desc.get("is_cylindrical", False)
    has_flats = desc.get("has_flat_surfaces", False)

    logger.info(f"Routing selection: cylindrical={is_cylindrical}, dim={largest_dim}mm, holes={holes}, complexity={complexity}")

    # Cylindrical parts
    if is_cylindrical:
        if largest_dim < 500:
            routings.append("CNC Lathe")
            reasons.append("Cylindrical geometry <500mm — CNC lathe preferred.")
        else:
            routings.append("Boring Mill")
            reasons.append("Large cylindrical part — boring mill required.")
        if holes > 0:
            routings.append("VMC Machining")
            reasons.append(f"{holes} hole(s) detected — secondary drilling on VMC.")
        if grooves > 0:
            routings.append("Keyway Machine")
            reasons.append(f"{grooves} groove(s) — keyway slotting required.")

    # Prismatic parts
    elif has_flats:
        if largest_dim < 1000:
            routings.append("VMC Machining")
            reasons.append("Prismatic geometry <1m — VMC suitable.")
        else:
            routings.append("Boring Mill")
            reasons.append("Large prismatic part — boring mill.")
        if holes > 0:
            routings.append("VMC Machining")
            reasons.append(f"{holes} hole(s) — drilling cycle on VMC.")
        if complexity >= 8:
            routings.append("Wire EDM")
            reasons.append("Complex geometry — Wire EDM for internal features.")

    # Material modifiers
    mat = material.lower()
    if any(m in mat for m in ["stainless", "hardened", "tool steel"]):
        routings.append("Wire EDM")
        reasons.append("Hard material — EDM required for precision.")
    if "aluminum" in mat:
        reasons.append("Aluminum — high-speed cutting suitable.")
    if "brass" in mat or "copper" in mat:
        reasons.append("Excellent machinability — faster feeds.")

    # Tolerance adjustments
    if tolerance < 0.01:
        routings.append("Wire EDM")
        reasons.append(f"Tight tolerance ±{tolerance}mm — EDM finishing.")
    elif tolerance < 0.02:
        reasons.append(f"Tolerance ±{tolerance}mm — fine finishing required.")

    if not routings:
        routings.append("VMC Machining")
        reasons.append("General machining fallback — VMC selected.")

    ordered = []
    [ordered.append(r) for r in routings if r not in ordered]
    return {"recommended_routings": ordered, "reasoning": reasons}


# =============================
# MACHINING ESTIMATION MODULE
# =============================

MRR_BY_PROCESS = {"VMC Machining": 15, "CNC Lathe": 20, "Boring Mill": 10, "Keyway Machine": 5, "Wire EDM": 0.8}
RATE_BY_PROCESS = {"VMC Machining": 80, "CNC Lathe": 75, "Boring Mill": 90, "Keyway Machine": 70, "Wire EDM": 120}
MATERIAL_FACTORS = {
    "aluminum": 0.8, "aluminium": 0.8, "brass": 0.9, "copper": 1.0,
    "mild steel": 1.0, "cold rolled steel": 1.1, "stainless": 1.3,
    "tool steel": 1.4, "hardened": 1.5, "titanium": 1.6
}

def estimate_machining_time_and_cost(desc, material, routings):
    """Estimate machining time (minutes) and cost ($USD)"""
    volume = desc.get("volume_cm3", 0)
    complexity = desc.get("complexity_score", 5)
    grooves = desc.get("grooves_count", 0)
    bbox = desc.get("bounding_box", [0, 0, 0])
    largest_dim = max(bbox) if bbox else 0
    mat_factor = 1.0
    for k, v in MATERIAL_FACTORS.items():
        if k in material.lower():
            mat_factor = v
            break
    complexity_factor = 1 + (complexity - 5) * 0.05

    total_cost, estimates = 0, []
    for r in routings:
        mrr = MRR_BY_PROCESS.get(r, 10)
        rate = RATE_BY_PROCESS.get(r, 75)
        if r == "Wire EDM":
            perimeter_cm = (bbox[0] + bbox[1]) / 10
            time_min = (perimeter_cm * 10) * mat_factor
        elif r == "Keyway Machine":
            time_min = (grooves * 15) * mat_factor
        else:
            time_min = (volume / mrr) * mat_factor * complexity_factor

        setup = 20 if largest_dim > 500 else 10
        total_time = time_min + setup
        cost = (total_time / 60) * rate
        total_cost += cost

        estimates.append({
            "routing": r,
            "machining_time_min": round(total_time, 2),
            "machining_cost_usd": round(cost, 2)
        })

    return {"machining_summary": estimates, "total_cost_usd": round(total_cost, 2)}


# =============================
# MESH GENERATION HELPERS
# =============================

def calculate_face_center(triangulation, transform):
    """Calculate centroid of face"""
    x_sum = y_sum = z_sum = 0
    count = triangulation.NbNodes()
    for i in range(1, count + 1):
        pnt = triangulation.Node(i)
        pnt.Transform(transform)
        x_sum += pnt.X()
        y_sum += pnt.Y()
        z_sum += pnt.Z()
    return [x_sum/count, y_sum/count, z_sum/count]

def get_average_face_normal(triangulation, transform, face_reversed):
    """Calculate average normal of face"""
    triangle = triangulation.Triangle(1)
    n1, n2, n3 = triangle.Get()
    
    p1 = triangulation.Node(n1)
    p2 = triangulation.Node(n2)
    p3 = triangulation.Node(n3)
    
    p1.Transform(transform)
    p2.Transform(transform)
    p3.Transform(transform)
    
    edge1 = [p2.X()-p1.X(), p2.Y()-p1.Y(), p2.Z()-p1.Z()]
    edge2 = [p3.X()-p1.X(), p3.Y()-p1.Y(), p3.Z()-p1.Z()]
    
    normal = [
        edge1[1]*edge2[2] - edge1[2]*edge2[1],
        edge1[2]*edge2[0] - edge1[0]*edge2[2],
        edge1[0]*edge2[1] - edge1[1]*edge2[0]
    ]
    
    length = math.sqrt(sum(n*n for n in normal))
    if length > 0:
        normal = [n/length for n in normal]
    if face_reversed:
        normal = [-n for n in normal]
    
    return normal

def tessellate_shape(shape, quality=0.5):
    """
    Tessellate STEP shape into triangulated mesh for 3D rendering with face classification
    
    Args:
        shape: TopoDS_Shape from STEP file
        quality: 0-1 value, higher = finer mesh (more triangles, slower)
    
    Returns:
        dict with vertices, indices, normals arrays + face classifications for rendering
    """
    try:
        # Quality controls deflection (lower deflection = finer mesh)
        deflection = 1.1 - quality  # Map to 0.01-0.6mm range
        angular_deflection = 0.2  # ~11.5 degrees
        
        logger.info(f"Tessellating shape with quality={quality} (deflection={deflection}mm)")
        mesh = BRepMesh_IncrementalMesh(shape, deflection, False, angular_deflection, True)
        mesh.Perform()
        
        if not mesh.IsDone():
            raise Exception("Mesh tessellation failed")
        
        # Get bounding box for classification
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        center_z = (zmin + zmax) / 2
        max_radius = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2
        
        vertices = []
        indices = []
        normals = []
        face_types = []
        vertex_map = {}
        current_index = 0
        
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        
        while face_explorer.More():
            face = face_explorer.Current()
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(face, location)
            
            if triangulation is None:
                face_explorer.Next()
                continue
            
            transform = location.Transformation()
            surface = BRepAdaptor_Surface(face)
            face_reversed = face.Orientation() == 1
            surface_type = surface.GetType()
            
            # Classify face type for color coding
            face_classification = 'external'
            face_center = calculate_face_center(triangulation, transform)
            vector_to_center = [
                center_x - face_center[0],
                center_y - face_center[1],
                center_z - face_center[2]
            ]
            normal_vec = get_average_face_normal(triangulation, transform, face_reversed)
            dot_product = sum(n * v for n, v in zip(normal_vec, vector_to_center))
            
            if surface_type == GeomAbs_Cylinder:
                cylinder = surface.Cylinder()
                cyl_radius = cylinder.Radius()
                if cyl_radius < max_radius * 0.4 or dot_product > 0.3:
                    face_classification = 'internal'
                else:
                    face_classification = 'cylindrical'
            elif surface_type == GeomAbs_Plane:
                if dot_product > 0.5:
                    face_classification = 'internal'
                else:
                    face_classification = 'planar'
            else:
                if dot_product > 0.3:
                    face_classification = 'internal'
                else:
                    face_classification = 'external'
            
            face_vertices = []
            
            # Extract vertices
            for i in range(1, triangulation.NbNodes() + 1):
                pnt = triangulation.Node(i)
                pnt.Transform(transform)
                coord = (round(pnt.X(), 6), round(pnt.Y(), 6), round(pnt.Z(), 6))
                
                if coord not in vertex_map:
                    vertices.extend([pnt.X(), pnt.Y(), pnt.Z()])
                    vertex_map[coord] = current_index
                    face_vertices.append(current_index)
                    current_index += 1
                else:
                    face_vertices.append(vertex_map[coord])
            
            # Extract triangles
            for i in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(i)
                n1, n2, n3 = triangle.Get()
                
                idx1 = face_vertices[n1 - 1]
                idx2 = face_vertices[n2 - 1]
                idx3 = face_vertices[n3 - 1]
                
                if face_reversed:
                    indices.extend([idx1, idx3, idx2])
                else:
                    indices.extend([idx1, idx2, idx3])
                
                # Calculate triangle normal
                v1 = vertices[idx1*3:idx1*3+3]
                v2 = vertices[idx2*3:idx2*3+3]
                v3 = vertices[idx3*3:idx3*3+3]
                
                edge1 = [v2[i] - v1[i] for i in range(3)]
                edge2 = [v3[i] - v1[i] for i in range(3)]
                
                normal = [
                    edge1[1] * edge2[2] - edge1[2] * edge2[1],
                    edge1[2] * edge2[0] - edge1[0] * edge2[2],
                    edge1[0] * edge2[1] - edge1[1] * edge2[0]
                ]
                
                length = math.sqrt(sum(n*n for n in normal))
                if length > 0:
                    normal = [n / length for n in normal]
                
                if face_reversed:
                    normal = [-n for n in normal]
                
                for _ in range(3):
                    normals.extend(normal)
                    face_types.append(face_classification)
            
            face_explorer.Next()
        
        triangle_count = len(indices) // 3
        logger.info(f"Tessellation complete: {len(vertices)//3} vertices, {triangle_count} triangles")
        
        return {
            'vertices': vertices,
            'indices': indices,
            'normals': normals,
            'face_types': face_types,
            'triangle_count': triangle_count
        }
        
    except Exception as e:
        logger.error(f"Error tessellating shape: {e}")
        return {
            'vertices': [],
            'indices': [],
            'normals': [],
            'triangle_count': 0
        }


# =============================
# MAIN CAD ANALYSIS ENDPOINT
# =============================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "industrial-cad-analyzer"}), 200

@app.route("/analyze-cad", methods=["POST"])
def analyze_cad():
    """Main endpoint: CAD upload → geometry + mesh + routing + cost"""
    if "file" not in request.files:
        return jsonify({"error": "No CAD file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    material = request.form.get("material", "Cold Rolled Steel")
    tolerance = float(request.form.get("tolerance", 0.02))
    quality = float(request.form.get("quality", 0.999))
    
    logger.info(f"Analyzing: {file.filename}, material={material}, tol={tolerance}mm, quality={quality}")

    # Save file temporarily
    ext = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        reader = STEPControl_Reader()
        if reader.ReadFile(tmp_path) != 1:
            raise Exception("Failed to read STEP/IGES file.")
        reader.TransferRoots()
        shape = reader.OneShape()
        if shape.IsNull():
            raise Exception("Invalid or empty CAD geometry.")

        # Volume, surface, bbox
        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)
        volume_mm3 = props.Mass()
        brepgprop.SurfaceProperties(shape, props)
        surf_area = props.Mass()
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        width, height, depth = xmax - xmin, ymax - ymin, zmax - zmin

        # Surface classification
        cyl_faces, planar_faces, total_faces = 0, 0, 0
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            s = BRepAdaptor_Surface(explorer.Current())
            if s.GetType() == GeomAbs_Cylinder:
                cyl_faces += 1
            elif s.GetType() == GeomAbs_Plane:
                planar_faces += 1
            total_faces += 1
            explorer.Next()

        is_cylindrical = cyl_faces > total_faces * 0.4 if total_faces else False
        has_flat = planar_faces > 2
        complexity = min(10, max(1, int(total_faces / 10) + 3))

        # Generate mesh data for 3D preview
        mesh_data = tessellate_shape(shape, quality)

        # Build geometry descriptor
        desc = {
            "bounding_box": [round(width,2), round(height,2), round(depth,2)],
            "volume_cm3": round(volume_mm3 / 1000, 2),
            "is_cylindrical": is_cylindrical,
            "has_flat_surfaces": has_flat,
            "holes_count": 0,  # Simplified for merged version
            "grooves_count": 0,  # Simplified for merged version
            "complexity_score": complexity,
            "tolerance": tolerance
        }

        # Routing selection + cost estimation
        routing_result = select_routings_industrial(desc, material)
        machining_est = estimate_machining_time_and_cost(desc, material, routing_result["recommended_routings"])

        # Build complete result with mesh data
        result = {
            "volume_cm3": desc["volume_cm3"],
            "surface_area_cm2": round(surf_area / 100, 2),
            "part_width_cm": round(width / 10, 2),
            "part_height_cm": round(height / 10, 2),
            "part_depth_cm": round(depth / 10, 2),
            "complexity_score": complexity,
            "is_cylindrical": is_cylindrical,
            "has_flat_surfaces": has_flat,
            "cylindrical_faces": cyl_faces,
            "planar_faces": planar_faces,
            "total_faces": total_faces,
            "confidence": 0.95,
            "method": "pythonOCC_geometry_parsing",
            "mesh_data": mesh_data,  # ← THIS IS THE KEY FOR 3D PREVIEW
            "recommended_routings": routing_result["recommended_routings"],
            "routing_reasoning": routing_result["reasoning"],
            "machining_summary": machining_est["machining_summary"],
            "estimated_total_cost_usd": machining_est["total_cost_usd"]
        }

        logger.info(f"Analysis complete: {len(mesh_data.get('vertices', []))//3} vertices, routings={routing_result['recommended_routings']}, cost=${machining_est['total_cost_usd']}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"CAD analysis failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    finally:
        try: os.unlink(tmp_path)
        except: pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
