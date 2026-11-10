"""The explicit, public example catalogue shared by the server and packager."""
from __future__ import annotations

CATALOGUE = [
    {"slug": "aero-engine", "title": "航空发动机", "category": "航空动力", "description": "风扇、压气机、燃烧段与涡轮", "reference": "https://www.geaerospace.com/propulsion/commercial"},
    {"slug": "airliner", "title": "飞机", "category": "航空运输", "description": "机身、机翼、尾翼与双发动机", "reference": "https://www.airbus.com/en/products-services/commercial-aircraft"},
    {"slug": "luxury-liner", "title": "豪华轮船", "category": "船舶工程", "description": "船体、分层甲板、泳池与救生艇", "reference": "https://www.royalcaribbean.com/cruise-ships"},
    {"slug": "model-y", "title": "Tesla Model Y", "category": "电动汽车", "description": "车身、电池、驱动与四轮组件", "reference": "https://www.tesla.com/modely"},
    {"slug": "iphone", "title": "iPhone 手机", "category": "消费电子", "description": "屏幕、中框、电池、主板与相机", "reference": "https://www.apple.com/iphone/"},
    {"slug": "optimus", "title": "Tesla 机器人", "category": "人形机器人", "description": "头部、躯干、关节与四肢", "reference": "https://www.tesla.com/AI"},
    {"slug": "falcon", "title": "猎鹰火箭", "category": "航天运载", "description": "两级箭体、整流罩与发动机组", "reference": "https://www.spacex.com/vehicles/falcon-9/"},
    {"slug": "mechanical-watch", "title": "机械手表 · 劳力士", "category": "精密机械", "description": "表壳、表圈、表盘与机芯示意", "reference": "https://www.rolex.com/watches"},
    {"slug": "refinery", "title": "炼油厂", "category": "工业设施", "description": "塔器、储罐、换热器与管廊", "reference": None},
    {"slug": "industrial-arm", "title": "工业机械臂", "category": "工业机器人", "description": "底座、肩肘关节、腕部与夹爪", "reference": None},
]

LEGACY_CAD = [
    {"slug": "drive", "title": "轴 / 轴承 / 齿轮", "category": "参数化 CAD", "description": "传动装配 · 可下载 STEP"},
    {"slug": "clamp", "title": "夹持机构", "category": "参数化 CAD", "description": "夹爪与导向 · 可下载 STEP"},
]


def public_catalogue():
    return [*CATALOGUE, *LEGACY_CAD]
