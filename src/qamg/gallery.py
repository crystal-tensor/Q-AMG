"""The explicit, public example catalogue shared by the server and packager."""
from __future__ import annotations

CATALOGUE = [
    {"slug": "aero-engine", "title": "航空发动机", "category": "航空动力", "description": "逐片叶片、转静子、双轴、火焰筒与附件", "reference": "https://www.grc.nasa.gov/www/k-12/airplane/turbfan.html"},
    {"slug": "airliner", "title": "飞机", "category": "航空运输", "description": "蒙皮、隔框、翼梁翼肋、操纵面与轮端", "reference": "https://www.faa.gov/documentlibrary/media/advisory_circular/ac_65-15a.pdf"},
    {"slug": "luxury-liner", "title": "豪华轮船", "category": "船舶工程", "description": "肋骨舱壁、发电机组、电力推进与救生装置", "reference": "https://www.wartsila.com/encyclopedia/term/diesel-electric-propulsion"},
    {"slug": "model-y", "title": "Tesla Model Y", "category": "电动汽车", "description": "车门座椅、驱动单元、四轮制动与悬架", "reference": "https://www.tesla.com/modely"},
    {"slug": "iphone", "title": "iPhone 手机", "category": "消费电子", "description": "16 Pro 维修总成、连接压板、接口与紧固件", "reference": "https://support.apple.com/en-us/120803"},
    {"slug": "optimus", "title": "Tesla 机器人", "category": "人形机器人", "description": "关节传动、逐指指节、骨架与电控推定", "reference": "https://www.tesla.com/SI"},
    {"slug": "falcon", "title": "猎鹰火箭", "category": "航天运载", "description": "逐台 Merlin、双推进剂箱、回收与载荷接口", "reference": "https://www.spacex.com/vehicles/falcon-9/"},
    {"slug": "mechanical-watch", "title": "机械手表 · 劳力士", "category": "精密机械", "description": "发条盒、轮系、擒纵、游丝、日历与逐节表链", "reference": "https://newsroom.rolex.com/watchmaking/features/movements"},
    {"slug": "refinery", "title": "炼油厂", "category": "工业设施", "description": "塔盘降液管、换热管束、离心泵与阀门", "reference": "https://www.eia.gov/kids/energy-sources/oil/"},
    {"slug": "industrial-arm", "title": "工业机械臂", "category": "工业机器人", "description": "六轴伺服、减速器、编码器、制动与夹爪", "reference": "https://legacy.harmonicdrive.net/reference/applicationnotes/principles.php"},
]

LEGACY_CAD = [
    {"slug": "drive", "title": "轴 / 轴承 / 齿轮", "category": "参数化 CAD", "description": "传动装配 · 可下载 STEP"},
    {"slug": "clamp", "title": "夹持机构", "category": "参数化 CAD", "description": "夹爪与导向 · 可下载 STEP"},
]


def public_catalogue():
    return [*CATALOGUE, *LEGACY_CAD]
