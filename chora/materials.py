"""Explicit, uncalibrated reduced-order material assumptions, in SI units.

Visual material and mechanical law are separate. Steel is rigid in this model;
Young's modulus/hardness do not silently alter MuJoCo's solver contact settings.
"""
import math

PINE={"name":"Pine · illustrative model","density_kg_m3":500.,"bearing_pressure_pa":5e6,
      "shank_shear_pa":3e4,"moisture_percent":12.,"board_thickness_m":.03,
      "calibrated":False,"grain":"normal to board face; no splitting model"}
STEEL={"name":"Steel · rigid body","density_kg_m3":7850.,"elastic_modulus_pa":200e9,
       "hardness":"not simulated; no plastic bending or edge wear"}
FRUIT={
 "apple":{"name":"Apple","radius":.023,"height":.040,"mass":.045,"skin_force_n":7.,"flesh_force_n":4.,"skin_thickness_m":.0004,"fracture_work_j":.13},
 "tomato":{"name":"Tomato","radius":.023,"height":.036,"mass":.040,"skin_force_n":4.,"flesh_force_n":1.3,"skin_thickness_m":.0002,"fracture_work_j":.055},
 "banana":{"name":"Banana · unpeeled","radius":.014,"height":.028,"mass":.035,"skin_force_n":5.,"flesh_force_n":1.8,"skin_thickness_m":.002,"fracture_work_j":.065},
}

def nail_resistance(depth,diameter=.002,wood=PINE):
    """Tip bearing + embedded-shank shear, not a nail-driving standard."""
    embedded=min(max(depth,0.),wood["board_thickness_m"])
    return wood["bearing_pressure_pa"]*math.pi*(diameter/2)**2 + wood["shank_shear_pa"]*math.pi*diameter*embedded


def cutting_resistance(kind,penetration):
    p=FRUIT[kind]
    if penetration<=0 or penetration>=p["height"]:return 0.
    # Skin puncture transitions to flesh resistance, with lower skin at exit.
    skin=penetration<p["skin_thickness_m"] or penetration>p["height"]-p["skin_thickness_m"]
    return p["skin_force_n"] if skin else p["flesh_force_n"]

# Cohesive-work threshold follows the integral of the illustrative layered law.
# The 1 mm terminal tolerance matches this reduced-order contact geometry.
for _food in FRUIT.values():
    _food['fracture_work_j']=.95*(2*_food['skin_force_n']*_food['skin_thickness_m']+_food['flesh_force_n']*(_food['height']-2*_food['skin_thickness_m']))
