import os
import io
import math
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client

# === OCC imports ===
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane

import logging

# === CONFIG ===
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# === Supabase setup ===
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------------------------------------
# === Geometry Utilities ===
# --------------------------------------------------

def extract_feature_edges(shape):
    """Safely extract feature edges with fallback if OCC fails"""
    edges = []
    try:
        explorer = TopExp_Explorer(shape, 1)  # 1 = EDGE
        while explorer.More():
            edge = explorer.Current()
            curve_handle, first, last = BRep_Tool.Curve(edge)
            if curve_handle is not None:
                p1 = curve_handle.Value(first)
                p2 = curve_handle.Value(last)
                edges.append([
                    [round(p1.X(), 4), round(p1.Y(), 4), round(p1.Z(), 4)],
                    [round(p2.X(), 4), round(p2.Y(), 4), round(p2.Z(), 4)]
                ])
            explorer.Next()
        logger.info(f"Extracted {len(edges)} feature edges safely")
    except Exception as e:
        logger.warning(f"Feature edge extraction failed: {e}")
        edges = []
    return edges


def calculate_face_center(triangulation, transform):
    """Compute the average center of a face from its triangulation"""
    try:
        total = np.zeros(3)
        for i in range(1, triangulation.NbNodes() + 1):
            p = triangulation.Node(i)
            p.Transform(transform)
            total += np.array([p.X(), p.Y(), p.Z()])
        return (total / triangulation.NbNodes()).tolist()
    except Exception:
        return [0, 0, 0]


def get_average_face_normal(triangulation, transform, reversed_face=False):
    """Approximate average normal of a face"""
    try:
        v1 = triangulation.Node(1)
        v2 = triangulation.Node(2)
        v3 = triangulation.Node(3)
        v1.Transform(transform)
        v2.Transform(transform)
        v3.Transform(transform)
        e1 = np.array([v2.X() - v1.X(), v2.Y() - v1.Y(), v2.Z() - v1.Z()])
        e2 = np.array([v3.X() - v1.X(), v3.Y() - v1.Y(), v3.Z() - v1.Z()])
        n = np.cross(e1, e2)
        n /= np.linalg.norm(n) + 1e-9
        if reversed_face:
            n *= -1
        return n.tolist()
    except Exception:
        return [0, 0, 1]


# --------------------------------------------------
# === Tessellation ===
# --------------------------------------------------

def tessellate_shape(shape, quality=0.5):
    """Generate vertices, normals, indices, and classify faces"""
    try:
        deflection = 0.6 * (10 ** (-(quality * 3)))
        angular_deflection = 0.04
        mesh = BRepMesh_IncrementalMesh(shape, deflection, False, angular_deflection, True)
        mesh.Perform()
        if not mesh.IsDone():
            raise Exception("Mesh tessellation failed")

        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
        max_radius = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2

        vertices, indices, normals, face_types = [], [], [], []
        vertex_map = {}
        current_index = 0
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)

        while face_explorer.More():
            face = face_explorer.Current()
            loc = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(face, loc)
            if triangulation is None:
                face_explorer.Next()
                continue

            transform = loc.Transformation()
            surface = BRepAdaptor_Surface(face)
            reversed_face = face.Orientation() == 1
            surf_type = surface.GetType()
            center = calculate_face_center(triangulation, transform)
            vector_to_center = [cx - center[0], cy - center[1], cz - center[2]]
            normal_vec = get_average_face_normal(triangulation, transform, reversed_face)
            dot = sum(n * v for n, v in zip(normal_vec, vector_to_center))
            dot /= (math.sqrt(sum(v * v for v in vector_to_center)) + 1e-9)

            # classify face
            if surf_type == GeomAbs_Cylinder:
                cyl = surface.Cylinder()
                r = cyl.Radius()
                if r < max_radius * 0.4 or dot > 0:
                    ftype = "internal"
                else:
                    ftype = "cylindrical"
            elif surf_type == GeomAbs_Plane:
                ftype = "internal" if dot > 0.5 else "planar"
            else:
                ftype = "internal" if dot > 0.3 else "external"

            # vertices
            face_vertices = []
            for i in range(1, triangulation.NbNodes() + 1):
                p = triangulation.Node(i)
                p.Transform(transform)
                coord = (round(p.X(), 6), round(p.Y(), 6), round(p.Z(), 6))
                if coord not in vertex_map:
                    vertex_map[coord] = current_index
                    vertices.extend(coord)
                    face_vertices.append(current_index)
                    current_index += 1
                else:
                    face_vertices.append(vertex_map[coord])

            # triangles
            for i in range(1, triangulation.NbTriangles() + 1):
                tri = triangulation.Triangle(i)
                n1, n2, n3 = tri.Get()
                idx = [face_vertices[n1 - 1], face_vertices[n2 - 1], face_vertices[n3 - 1]]
                if reversed_face:
                    indices.extend([idx[0], idx[2], idx[1]])
                else:
                    indices.extend(idx)
                v1, v2, v3 = [vertices[j * 3:j * 3 + 3] for j in idx]
                e1 = [v2[k] - v1[k] for k in range(3)]
                e2 = [v3[k] - v1[k] for k in range(3)]
                n = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ]
                l = math.sqrt(sum(x * x for x in n))
                if l > 0:
                    n = [x / l for x in n]
                if sum(n * v for n, v in zip(n, vector_to_center)) < 0:
                    n = [-x for x in n]
                for _ in range(3):
                    normals.extend(n)
                    face_types.append(ftype)

            face_explorer.Next()

        triangle_count = len(indices) // 3
        feature_edges = extract_feature_edges(shape)
        logger.info(
            f"Tessellation complete: {len(vertices)//3} vertices, {triangle_count} triangles, {len(feature_edges)} edges"
        )
        return {
            "vertices": vertices,
            "indices": indices,
            "normals": normals,
            "face_types": face_types,
            "triangle_count": triangle_count,
            "feature_edges": feature_edges,
        }

    except Exception as e:
        logger.error(f"Tessellation error: {e}")
        return {
            "vertices": [],
            "indices": [],
            "normals": [],
            "face_types": [],
            "triangle_count": 0,
            "feature_edges": [],
        }

# --------------------------------------------------
# === STEP file upload endpoint ===
# --------------------------------------------------

@app.route("/analyze-cad", methods=["POST"])
def analyze_cad():
    """Upload a STEP file, mesh it, store in Supabase, return data"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if not file.filename.lower().endswith(".step"):
            return jsonify({"error": "Only .step files are supported"}), 400

        step_bytes = io.BytesIO(file.read())

        reader = STEPControl_Reader()
        status = reader.ReadFile(file.filename)
        if status != 1:
            return jsonify({"error": "Failed to read STEP file"}), 400
        reader.TransferRoots()
        shape = reader.OneShape()

        quality = float(request.form.get("quality", 0.5))
        mesh = tessellate_shape(shape, quality)
        feature_edges = extract_feature_edges(shape)

        mesh["feature_edges"] = feature_edges
        mesh["triangle_count"] = len(mesh.get("indices", [])) // 3

        # Save to Supabase
        data_insert = {
            "file_name": file.filename,
            "vertices": mesh["vertices"],
            "indices": mesh["indices"],
            "normals": mesh["normals"],
            "triangle_count": mesh["triangle_count"],
            "feature_edges": mesh["feature_edges"],
        }
        supabase.table("cad_meshes").insert(data_insert).execute()

        return jsonify(
            {
                "status": "success",
                "message": f"Mesh generated for {file.filename}",
                "mesh_data": mesh,
            }
        )

    except Exception as e:
        logger.error(f"Error processing CAD: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
