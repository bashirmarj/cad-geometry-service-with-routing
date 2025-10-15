"""
Unified CAD Geometry Analysis and Industrial Machining Quotation System
Parses STEP/IGES files, detects geometry, selects machining routings, and estimates machining cost.
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
# MAIN CAD ANALYSIS ENDPOINT
# =============================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "industrial-cad-analyzer"}), 200

@app.route("/analyze-cad", methods=["POST"])
def analyze_cad():
    """Main endpoint: CAD upload → routing + cost"""
    if "file" not in request.files:
        return jsonify({"error": "No CAD file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    material = request.form.get("material", "Cold Rolled Steel")
    tolerance = float(request.form.get("tolerance", 0.02))
    logger.info(f"Analyzing: {file.filename}, material={material}, tol={tolerance}")

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

        desc = {
            "bounding_box": [round(width,2), round(height,2), round(depth,2)],
            "volume_cm3": round(volume_mm3 / 1000, 2),
            "is_cylindrical": is_cylindrical,
            "has_flat_surfaces": has_flat,
            "holes_count": 0,
            "grooves_count": 0,
            "complexity_score": complexity,
            "tolerance": tolerance
        }

        # Routing selection + cost estimation
        routing_result = select_routings_industrial(desc, material)
        machining_est = estimate_machining_time_and_cost(desc, material, routing_result["recommended_routings"])

        result = {
            "geometry_descriptor": desc,
            "recommended_routings": routing_result["recommended_routings"],
            "routing_reasoning": routing_result["reasoning"],
            "machining_summary": machining_est["machining_summary"],
            "estimated_total_cost_usd": machining_est["total_cost_usd"]
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"CAD analysis failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    finally:
        try: os.unlink(tmp_path)
        except: pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
