from os import sep

# path to compressed save file
# change this to point to current game installation directory
INPUT_FILE_PATH = "/PATH/TO/YOUR/STEAM/" + \
                  "steamapps/common/Dungeon Defenders/Binaries/Win32/DunDefHeroes.dun".replace("/", sep)

"""
Set your desired weights for every character whose gear you want to optimize here.
Stats with no weights assigned to them will be set to 0.
The order of characters in OPTIM_TARGETS will determine their priority for item assignment.
Names for stats:
Hero health:    HHP
Hero damage:    HDMG
Hero speed:     HSPD
Hero repair:    HREP
Ability 1:      AB1
Ability 2::     AB2
Tower health:   THP
Tower damage:   TDMG
Tower rate:     TSPD
Tower range:    TRAN

The following is an example setup, don't forget to change the names to your actual character names
"""
tower_mage_weights = {"THP": 0.05, "TSPD": 0.9, "TDMG": 1, "TRAN": 0.25}
tower_monk_weights = {"THP": 0.5, "TSPD": 0.3, "TDMG": 0.95, "TRAN": 1}
dps_monk_weights = {"HHP": 0.9, "HDMG": 1, "AB1": 0.5, "AB2": 0.9}
dps_ev_weights = {"HHP": 0.5, "HDMG": 1, "AB2": 0.9}
tower_sum_weights = {"THP": 1, "TDMG": 0.9, "TSPD": 0.25, "TRAN": 0.15}
tower_hunter_weights = {"THP": 0.3, "TDMG": 0.9, "TRAN": 1, "TSPD": 0.3}
tower_ev_weights = {"THP": 0.7, "TDMG": 1, "TSPD": 0.5, "TRAN": 0.2}
tower_squire_weights = {"THP": 0.1, "TDMG": 1, "TSPD": 0.8, "TRAN": 0.25}
jester_weights = {"HHP": 0.7, "HDMG": 0.5, "HREP": 1, "AB2": 0.7}
buff_monk_weights = {"HHP": 0.4, "HDMG": 0.3, "AB1": 1}
barbarian_weights = {"HHP": 0.5, "HDMG": 1, "AB2": 0.9}
support_sum_weights = {"HHP": 1, "AB2": 0.5, "HREP": 0.5}
buff_sum_weights = {"HHP": 1, "AB2": 0.1}
OPTIM_TARGETS = {
    "BUILDER_MAGE_NAME": tower_mage_weights,
    "BUILDER_MONK_NAME": tower_monk_weights,
    "BUILDER_SUMMONER_NAME": tower_sum_weights,
    "BUILDER_EV_NAME": tower_ev_weights,
    "DPS_EV_NAME": dps_ev_weights,
    "DPS_MONK_NAME": dps_monk_weights,
    "BUILDER_HUNTER_NAME": tower_hunter_weights,
    "JESTER_NAME": jester_weights,
    "AB2_MONK_NAME": buff_monk_weights,
    "BARBARIAN_NAME": barbarian_weights,
    "BUILDER_SQUIRE_NAME": tower_squire_weights,
    "AB2_SUMMONER_NAME": support_sum_weights,
    "BUFF_SUMMONER_NAME_1": buff_sum_weights,
    "BUFF_SUMMONER_NAME_2": buff_sum_weights
}

# modifier keys used for stacking towers, pick any combination of [ctrl, shift, alt]
STACK_MODIFIER_KEYS = ["ctrl"]
# set these to all keys used in game to build towers
STACK_TOWER_KEYS = ["6", "7", "8", "9", "0"]
# sends two mouse clicks when pressed together with STACK_MODIFIER_KEYS
STACK_MINION_KEY = "l"
# sends right mouse down after when pressed together with STACK_MODIFIER_KEYS
HOLD_MOUSE_KEY = "j"
# modifier keys used to quit the application, pick any combination of [ctrl, shift, alt]
QUIT_MODIFIER_KEYS = ["ctrl", "shift"]
# quits the application when pressed together with QUIT_MODIFIER_KEYS
QUIT_KEY = "e"
