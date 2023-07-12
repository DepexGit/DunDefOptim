from os import sep
from collections import defaultdict

# path to decompressed save file
DECOMP_FILE_PATH = "data/DunDefHeroes.decomp".replace("/", sep)
# path to file storing pickle dumb of pre initialized EquipmentHandler
EQUIP_FILE_PATH = "data/equip.bin".replace("/", sep)
# maps quality ids to set bonus, max stat bonus and quality name
QUALITY_PROPERTIES = defaultdict(lambda: [1.25, 360, "Low"], 
        {13: [1.3, 360, "Myth"],
         14: [1.33, 420, "Trans"],
         15: [1.36, 500, "Sup"],
         16: [1.4, 600, "Ult"],
         17: [1.4, 600, "Ult93"],
         18: [1.4, 700, "Ult+"],
         19: [1.4, 999, "Ult++"]})
# maps class ids to their names
CLASS_NAMES = {
    "Recruit": "Monk",
    "Monkette": "Monk",
    "Apprentice": "Mage",
    "Sorceress": "Mage",
    "Squire": "Squire",
    "LadyKnight": "Squire",
    "Initiate": "Hunter",
    "Hunter": "Hunter",
    "RobotGirl": "SeriesEV",
    "Jester": "Jester",
    "Summoner": "Summoner",
    "Barbarian": "Barbarian",
    "Hermit": "Hermit"}
# maps weapon class name to possible weapon id strings
WEAPON_NAMES = {
    "Monk": ["Spear", "CardboardTube", "Weapons.Campaign.Monk", "LavaChakram", "ChickenBallBlaster",
             "TurtlePoleArm"],
    "Mage": ["Staff", "Sceptre", "Weapon.Apprentice", "Weapons.Challenge.Apprentice", "NorthPole",
             "Weapons.Sandstorm"],
    "Squire": ["Broadsword", "SquireSword", "Weapons.Campaign.Squire", "Weapons.Squire",
               "IceBlade", "Equipment.Squire", "SquireEquipment", "Squire_Sword"],
    "Hunter": ["Crossbow", "Weapon.Huntress", "ForestBow", "Equipment.Huntress",
               "LampGun", "portalGun", "Flamethrower", "WaterGun", "HuntressEquipment"]}
# maps equipment types to possible id strings
EQUIPMENT_TYPES = {
    "Armor": ["DunDefEquipment.EquipmentGeneric."],
    "Accessory": ["AccessoryEquipment", "AccesoryEquipment"],
    "Familiar": ["Familiar", "familiar", "SkyCityPet", "PetRock", "FamGuardians"],
    "Weapon": [e for sl in WEAPON_NAMES.values() for e in sl]}
# contains slots for armors
ARMOR_SLOTS = ["Helmet", "Torso", "Gauntlet", "Boots"]
# maps accessory slots to possible id strings
ACCESSORY_SLOTS = {
    "Hat": ["Hat", "Bow", "Brooch", "HeadDress", "Feather", "Quill", "Item0", "ElfCrown"],
    "Mask": ["Beard", "BunnyWhiskers", "Mask", "Glasses", "PiratePatch", "Nose"],
    "Bracers": ["Bracer"],
    "Shield": ["Buckler", "Shield"]}
# maps character class names to equipment slots
CHAR_SLOTS = {
    "Mage": [["Mage"], ["Pet"]],
    "Monk": [["Monk"], ["Pet"]],
    "Squire": [["Squire"], ["Pet"]],
    "Hunter": [["Hunter"], ["Pet"]],
    "Barbarian": [["Squire"], ["Squire"], ["Pet"]],
    "Jester": [["Squire", "Hunter", "Monk", "Mage"], ["Squire", "Hunter", "Monk", "Mage"], ["Pet"]],
    "SeriesEV": [["Hunter", "Mage"], ["Hunter", "Mage"], ["Pet"]],
    "Summoner": [["Pet"], ["Pet"]],
    "Hermit": [["Monk"], ["Pet"]]}
# contains every armor material
ARMOR_MATERIALS = ["Leather", "Mail", "Pristine", "Chain", "Plate"]
# bytes prefix used to indentify equipment in save file
EQUIPMENT_MARKER = [1, 0, 1, 2, 3]
# maps stat types to offset and length after EQUIPMENT_MARKER
STAT_OFFSET_DICT = {
    "PHRES": [0, 1],
    "PORES": [1, 1],
    "HERES": [2, 1],
    "LIRES": [3, 1],
    "HHP": [8, 4],
    "HSPD": [12, 4],
    "HDMG": [16, 4],
    "HREP": [20, 4],
    "AB1": [24, 4],
    "AB2": [28, 4],
    "THP": [32, 4],
    "TSPD": [36, 4],
    "TDMG": [40, 4],
    "TRAN": [44, 4]}
# maps misc stats to offset and length after EQUIPMENT_MARKER
MISC_OFFSET_DICT = {
    "level": [114, 4],
    "quality": [145, 1],
    "max_upgrades": [164, 4],
    "pos_x": [168, 4],
    "pos_y": [172, 4],
    "pos_z": [176, 4],
    "name": [220, 4]}
# values used to calculate equipment upgrade costs
UPGRADE_COST_BASE = 3646.48407
UPGRADE_COST_EXP = 1.58700072
ACC_UPGRADE_COST = 17.9633e9
