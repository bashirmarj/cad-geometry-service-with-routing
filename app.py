"""
CAD Geometry Analysis Microservice
Parses STEP/IGES files using pythonOCC to extract real geometry data
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
import logging
import math

# OCC imports for STEP parsing
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
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

# -------------------------------------------------------------------------
# INLINE ROUTING + COST LOGIC (replaces missing modules)
# -------------------------------------------------------------------------
def select_routings_industrial(geometry_descriptor, material):
    """Simple inline industrial routing selector."""
    volume = geometry_descriptor.get("volume_cm3", 0)
    is_cylindrical = geometry_descriptor.get("is_cylindrical", False)
    holes = geometry_descriptor.get("holes_count", 0)
    grooves = geometry_descriptor.get("grooves_count", 0)
    complexity = geometry_descriptor.get("complexity_score", 5)

    if is_cylindrical and holes <= 2:
        routings = ["Lathe"]
        reasoning = "Cylindrical part with simple features → Lathe machining."
    elif is_cylindrical and grooves > 0:
        routings = ["Mill-Turn"]
        reasoning = "Cylindrical with grooves → Mill-Turn machining."
    elif not is_cylindrical and complexity <= 5:
        routings = ["VMC 3-axis"]
        reasoning = "Prismatic part, moderate complexity → VMC 3-axis machining."
    elif not is_cylindrical and complexity > 5:
        routings = ["HMC 4-axis"]
        reasoning = "Complex prismatic geometry → HMC 4-axis machining."
    else:
        routings = ["VMC 3-axis"]
        reasoning = "Default routing selection."

    logger.info(f"Routing decision: {routings} | {reasoning}")
    return {"recommended_routings": routings, "reasoning": reasoning}


def estimate_machining_time_and_cost(geometry_descriptor, material, routings):
    """Inline machining time & cost estimation."""
    base_rate = 60.0  # $/hr
    volume_cm3 = geometry_descriptor.get("volume_cm3", 0)
    complexity = geometry_descriptor.get("complexity_score", 5)
    routing_type = routings[0] if routings else "VMC 3-axis"

    machining_time_hr = (volume_cm3 / 500.0) * (complexity / 5.0)
    machine_factor = {"Lathe": 0.8, "Mill-Turn": 1.1, "HMC 4-axis": 1.3, "VMC 3-axis": 1.0}.get(routing_type, 1.0)
    machining_time_hr *= machine_factor

    total_cost = machining_time_hr * base_rate
    summary = f"Estimated machining time {machining_time_hr:.2f} hr using {routing_type} on {material}."

    logger.info(f"Machining estimate: {summary} | Cost: ${total_cost:.2f}")
    return {"machining_summary": summary, "total_cost_usd": round(total_cost, 2)}
# -------------------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'cad-geometry-analyzer'}), 200

@app.route('/analyze-cad', methods=['POST'])
def analyze_cad():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    material = request.form.get('material', 'Cold Rolled Steel')
    tolerance = float(request.form.get('tolerance', 0.02))
    quality = float(request.form.get('quality', 0.999))

    logger.info(f"Analyzing file: {file.filename}, material={material}, tol={tolerance}mm")

    # Save temp file
    file_ext = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        reader = STEPControl_Reader()
        status = reader.ReadFile(tmp_path)
        if status != 1:
            raise Exception(f"Failed to read STEP file (status={status})")

        reader.TransferRoots()
        shape = reader.OneShape()
        if shape.IsNull():
            raise Exception("No shape found in STEP file")

        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)
        volume_mm3 = props.Mass()

        brepgprop.SurfaceProperties(shape, props)
        surface_area_mm2 = props.Mass()

        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        width, height, depth = xmax - xmin, ymax - ymin, zmax - zmin

        # Face analysis
        cylindrical_faces = planar_faces = total_faces = 0
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            surface = BRepAdaptor_Surface(face)
            stype = surface.GetType()
            if stype == GeomAbs_Cylinder:
                cylindrical_faces += 1
            elif stype == GeomAbs_Plane:
                planar_faces += 1
            total_faces += 1
            explorer.Next()

        is_cylindrical = cylindrical_faces > (total_faces * 0.4)
        has_flat_surfaces = planar_faces > 2
        complexity = min(10, max(1, int(total_faces / 10) + 3))

        # Tessellation
        mesh_data = tessellate_shape(shape, quality)

        geometry_descriptor = {
            "volume_cm3": round(volume_mm3 / 1000, 2),
            "bounding_box": [width, height, depth],
            "is_cylindrical": is_cylindrical,
            "has_flat_surfaces": has_flat_surfaces,
            "holes_count": 0,
            "grooves_count": 0,
            "complexity_score": complexity,
            "tolerance": tolerance,
        }

        routing_result = select_routings_industrial(geometry_descriptor, material)
        machining_estimate = estimate_machining_time_and_cost(
            geometry_descriptor, material, routing_result["recommended_routings"]
        )

        result = {
            "volume_cm3": round(volume_mm3 / 1000, 2),
            "surface_area_cm2": round(surface_area_mm2 / 100, 2),
            "part_width_cm": round(width / 10, 2),
            "part_height_cm": round(height / 10, 2),
            "part_depth_cm": round(depth / 10, 2),
            "complexity_score": complexity,
            "is_cylindrical": is_cylindrical,
            "has_flat_surfaces": has_flat_surfaces,
            "cylindrical_faces": cylindrical_faces,
            "planar_faces": planar_faces,
            "total_faces": total_faces,
            "mesh_data": mesh_data,
            "recommended_routings": routing_result["recommended_routings"],
            "routing_reasoning": routing_result["reasoning"],
            "machining_summary": machining_estimate["machining_summary"],
            "estimated_total_cost_usd": machining_estimate["total_cost_usd"],
        }

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
# Tessellation Logic
# -------------------------------------------------------------------------
def calculate_face_center(triangulation, transform):
    x_sum = y_sum = z_sum = 0
    count = triangulation.NbNodes()
    for i in range(1, count + 1):
        pnt = triangulation.Node(i)
        pnt.Transform(transform)
        x_sum += pnt.X()
        y_sum += pnt.Y()
        z_sum += pnt.Z()
    return [x_sum / count, y_sum / count, z_sum / count]

def get_average_face_normal(triangulation, transform, face_reversed):
    tri = triangulation.Triangle(1)
    n1, n2, n3 = tri.Get()
    p1, p2, p3 = triangulation.Node(n1), triangulation.Node(n2), triangulation.Node(n3)
    p1.Transform(transform); p2.Transform(transform); p3.Transform(transform)
    e1 = [p2.X()-p1.X(), p2.Y()-p1.Y(), p2.Z()-p1.Z()]
    e2 = [p3.X()-p1.X(), p3.Y()-p1.Y(), p3.Z()-p1.Z()]
    normal = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]]
    l = math.sqrt(sum(n*n for n in normal))
    if l > 0: normal = [n/l for n in normal]
    if face_reversed: normal = [-n for n in normal]
    return normal

def tessellate_shape(shape, quality=0.999):
    """High-fidelity adaptive tessellation."""
    try:
        deflection = 0.6 * (10 ** (-(quality * 3)))  # logarithmic precision scale
        angular_deflection = 0.04  # radians (~2.3°)
        logger.info(f"Tessellating with deflection={deflection:.4f}mm, angular={angular_deflection:.3f}rad")
        mesh = BRepMesh_IncrementalMesh(shape, deflection, False, angular_deflection, True)
        mesh.Perform()
        if not mesh.IsDone():
            raise Exception("Mesh failed")

        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        cx, cy, cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2
        max_radius = max(xmax-xmin, ymax-ymin, zmax-zmin)/2

        vertices, indices, normals, face_types = [], [], [], []
        vertex_map, current_index = {}, 0

        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            face = face_explorer.Current()
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(face, location)
            if triangulation is None:
                face_explorer.Next(); continue
            transform = location.Transformation()
            surface = BRepAdaptor_Surface(face)
            face_reversed = face.Orientation() == 1
            surface_type = surface.GetType()

            fc = "external"
            face_center = calculate_face_center(triangulation, transform)
            to_center = [cx-face_center[0], cy-face_center[1], cz-face_center[2]]
            normal_vec = get_average_face_normal(triangulation, transform, face_reversed)
            dp = sum(n*v for n,v in zip(normal_vec,to_center))
            dp /= (math.sqrt(sum(v*v for v in to_center))+1e-9)

            if surface_type == GeomAbs_Cylinder:
                cyl_radius = surface.Cylinder().Radius()
                if cyl_radius < max_radius*0.4 or dp > 0: fc="internal"
                else: fc="cylindrical"
            elif surface_type == GeomAbs_Plane:
                fc = "internal" if dp>0.5 else "planar"
            else:
                fc = "internal" if dp>0.3 else "external"

            face_vertices=[]
            for i in range(1, triangulation.NbNodes()+1):
                p=triangulation.Node(i); p.Transform(transform)
                coord=(round(p.X(),6),round(p.Y(),6),round(p.Z(),6))
                if coord not in vertex_map:
                    vertices.extend([p.X(),p.Y(),p.Z()])
                    vertex_map[coord]=current_index
                    face_vertices.append(current_index)
                    current_index+=1
                else: face_vertices.append(vertex_map[coord])

            for i in range(1, triangulation.NbTriangles()+1):
                t=triangulation.Triangle(i); n1,n2,n3=t.Get()
                i1,i2,i3=face_vertices[n1-1],face_vertices[n2-1],face_vertices[n3-1]
                if face_reversed: indices.extend([i1,i3,i2])
                else: indices.extend([i1,i2,i3])
                v1=vertices[i1*3:i1*3+3]; v2=vertices[i2*3:i2*3+3]; v3=vertices[i3*3:i3*3+3]
                e1=[v2[i]-v1[i] for i in range(3)]; e2=[v3[i]-v1[i] for i in range(3)]
                n=[e1[1]*e2[2]-e1[2]*e2[1],e1[2]*e2[0]-e1[0]*e2[2],e1[0]*e2[1]-e1[1]*e2[0]]
                l=math.sqrt(sum(x*x for x in n))
                if l>0: n=[x/l for x in n]
                dp2=sum(nn*v for nn,v in zip(n,to_center))
                if dp2<0: n=[-x for x in n]
                for _ in range(3): normals.extend(n); face_types.append(fc)
            face_explorer.Next()

        logger.info(f"Tessellation complete: {len(vertices)//3} vertices, {len(indices)//3} triangles")
        return {"vertices": vertices, "indices": indices, "normals": normals, "face_types": face_types}
    except Exception as e:
        logger.error(f"Tessellation error: {e}")
        return {"vertices": [], "indices": [], "normals": [], "face_types": []}
# -------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
