"""
Mock wind-farm database for the chat demo (Xinjiang & Zhejiang).

All values are plausible but fictional, for demonstration only. Coordinates lie
within the model's Xinjiang / Zhejiang evaluation regions so the pipeline can
crop and downscale around each farm.
"""

WIND_FARMS = {
    # ---------------- Xinjiang (arid plateau, strong winds) ----------------
    "新疆十二间房风电场": {
        "id": "XJ-SSJF",
        "region": "新疆 (Xinjiang)",
        "lat": 43.05, "lon": 92.30,
        "elevation_m": 1280,           # 哈密戈壁台地
        "capacity_mw": 600.0,
        "turbines": 200,
        "hub_height_m": 100,
        "rotor_diameter_m": 121,
        "turbine_model": "GW3.0-121",
        "cut_in": 3.0, "rated": 11.5, "cut_out": 25.0,
        "terrain": "戈壁台地，开阔平坦，风能资源丰富（年均风速 ~7.5 m/s）",
    },
    "新疆达坂城风电场": {
        "id": "XJ-DBC",
        "region": "新疆 (Xinjiang)",
        "lat": 43.36, "lon": 88.31,
        "elevation_m": 1100,
        "capacity_mw": 500.0,
        "turbines": 250,
        "hub_height_m": 90,
        "rotor_diameter_m": 110,
        "turbine_model": "GW2.5-110",
        "cut_in": 3.0, "rated": 11.0, "cut_out": 25.0,
        "terrain": "达坂城风区，峡谷狭管效应显著，大风频发",
    },
    "新疆小草湖风电场": {
        "id": "XJ-XCH",
        "region": "新疆 (Xinjiang)",
        "lat": 43.20, "lon": 90.10,
        "elevation_m": 1000,
        "capacity_mw": 400.0,
        "turbines": 160,
        "hub_height_m": 100,
        "rotor_diameter_m": 121,
        "turbine_model": "GW3.0-121",
        "cut_in": 3.0, "rated": 11.5, "cut_out": 25.0,
        "terrain": "百里风区，地形开阔，强风时段集中",
    },
    "新疆三塘湖风电场": {
        "id": "XJ-STH",
        "region": "新疆 (Xinjiang)",
        "lat": 44.10, "lon": 92.80,
        "elevation_m": 1650,
        "capacity_mw": 300.0,
        "turbines": 100,
        "hub_height_m": 110,
        "rotor_diameter_m": 140,
        "turbine_model": "GW4.0-140",
        "cut_in": 2.8, "rated": 10.5, "cut_out": 24.0,
        "terrain": "高海拔盆地边缘，空气密度偏低",
    },
    # ---------------- Zhejiang (coastal / mountainous) ----------------
    "浙江括苍山风电场": {
        "id": "ZJ-KCS",
        "region": "浙江 (Zhejiang)",
        "lat": 28.62, "lon": 120.90,
        "elevation_m": 1380,           # 山地风电
        "capacity_mw": 120.0,
        "turbines": 60,
        "hub_height_m": 90,
        "rotor_diameter_m": 99,
        "turbine_model": "SE-9320",
        "cut_in": 3.0, "rated": 11.0, "cut_out": 22.0,
        "terrain": "浙东南山地，地形复杂，湍流强度较高",
    },
    "浙江大陈岛海上风电场": {
        "id": "ZJ-DCD",
        "region": "浙江 (Zhejiang)",
        "lat": 28.45, "lon": 121.88,
        "elevation_m": 0,              # 海上
        "capacity_mw": 200.0,
        "turbines": 40,
        "hub_height_m": 110,
        "rotor_diameter_m": 158,
        "turbine_model": "GW5.0-158",
        "cut_in": 3.0, "rated": 10.5, "cut_out": 25.0,
        "terrain": "近海海上风电，海面粗糙度低，风速稳定",
    },
    "浙江长龙山风电场": {
        "id": "ZJ-CLS",
        "region": "浙江 (Zhejiang)",
        "lat": 30.50, "lon": 119.65,
        "elevation_m": 1100,
        "capacity_mw": 90.0,
        "turbines": 45,
        "hub_height_m": 85,
        "rotor_diameter_m": 93,
        "turbine_model": "SE-9320",
        "cut_in": 3.0, "rated": 11.0, "cut_out": 22.0,
        "terrain": "浙西北山脊，受季风与地形抬升共同影响",
    },
    "浙江苍南海上风电场": {
        "id": "ZJ-CN",
        "region": "浙江 (Zhejiang)",
        "lat": 27.40, "lon": 120.95,
        "elevation_m": 0,
        "capacity_mw": 400.0,
        "turbines": 73,
        "hub_height_m": 100,
        "rotor_diameter_m": 152,
        "turbine_model": "MySE5.5-155",
        "cut_in": 3.0, "rated": 10.0, "cut_out": 25.0,
        "terrain": "浙南近海，台风季需关注切出风速",
    },
}


def list_farms():
    return list(WIND_FARMS.keys())


def find_farm(query: str):
    """Fuzzy-match a farm by name fragment. Returns (name, info) or (None, None)."""
    if not query:
        return None, None
    q = query.strip()
    # exact
    if q in WIND_FARMS:
        return q, WIND_FARMS[q]
    # substring on the distinctive part (strip 省/风电场)
    for name, info in WIND_FARMS.items():
        core = name.replace("风电场", "").replace("新疆", "").replace("浙江", "")
        if core and core in q:
            return name, info
        if info["id"].lower() in q.lower():
            return name, info
    # any token overlap
    for name, info in WIND_FARMS.items():
        if any(tok and tok in q for tok in [name[:4], name.replace("风电场", "")]):
            return name, info
    return None, None


def farms_brief():
    """Short catalogue string for the assistant's system context."""
    lines = []
    for name, f in WIND_FARMS.items():
        lines.append(
            f"- {name}（{f['id']}）：{f['region'].split(' ')[0]}，"
            f"({f['lat']:.2f}N,{f['lon']:.2f}E)，海拔{f['elevation_m']}m，"
            f"装机{f['capacity_mw']:.0f}MW，{f['turbines']}台×{f['turbine_model']}，"
            f"轮毂{f['hub_height_m']}m"
        )
    return "\n".join(lines)
