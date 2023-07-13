import numpy as np
import copy
import math
from prettytable import PrettyTable
from sys import float_info
from src.consts import *


class Equipment:
    """
    Holds all necessary information for an instance equipment and provides methods to estimate
    its value for optimization
    """

    def __init__(self, equip_type, equip_slot, armor_material=None):
        """
        Constructor for Equipment class.

        Parameters:
            equip_type (string): Equipment type, e.g. Armor
            equip_slot (string): Equipment slot, e.g. Helmet
            armor_material (string): Armor material, e.g. Chain
        """
        self.type = equip_type
        self.slot = equip_slot
        self.material = armor_material
        self.index = 0
        self.owner = ""
        self.dir_id = -1
        self.remaining_upgrades = 0
        self.remaining_upgrades_res = 0
        self.max_upgrades = 0
        self.level = 0
        self.stat_dict = {}
        self.quality = 0
        self.desc_string = ""
        self.pos_x = 0
        self.pos_y = 0
        self.pos_z = 0

    def init_upgrades(self):
        """
        Calculates available upgrades, assumes non builders always max resistances on armor.
        Must be called after initializing max_upgrades and level
        """
        self.remaining_upgrades = self.max_upgrades - self.level
        self.remaining_upgrades_res = self.remaining_upgrades
        resistances = list(self.stat_dict.values())[:4]
        res_target = max(0, 16 - self.quality) + 29
        # assume pre upgraded items are good
        if not (min(resistances) > 20 and self.level > 100):
            # simulate capping resistances
            for res in resistances:
                # missing resistance -> useless
                if res == 0:
                    self.remaining_upgrades_res = -1
                while res < res_target:
                    if res == 20 or res == 21:
                        res_offset = 3
                    elif 14 <= res < 20 or res == -1:
                        res_offset = 2
                    else:
                        res_offset = 1
                    res += res_offset
                    self.remaining_upgrades_res -= 1
                    # cannot max resistances -> useless
                    if self.remaining_upgrades_res == 0 and min(resistances) < res_target:
                        self.remaining_upgrades_res = -1

    def get_property_list(self, dirs=None, include_res=False):
        """
        Returns a list of item stats and properties

        Parameters:
            dirs (dict{int: list[string, int]}) Maps directory id to its name and parent id
            include_res (bool): If True: includes restistance values

        Returns:
            list[int]: List of equipment stats
        """
        location = self.owner
        cur_dir_id = self.dir_id
        stats = list(self.stat_dict.values())
        if not include_res:
            stats = stats[4:]
        all_vals = [self.type, self.slot, self.material,
                    QUALITY_PROPERTIES[self.quality][2]] + stats + \
                   [str(self.level) + "/" + str(self.max_upgrades)]
        if dirs is None:
            return all_vals
        if cur_dir_id != -1:
            path = []
            # check for root dir
            while cur_dir_id in dirs and cur_dir_id != dirs[cur_dir_id][1]:
                path.append(dirs[cur_dir_id][0])
                cur_dir_id = dirs[cur_dir_id][1]
            path = path[::-1]
            if len(path) == 0:
                location = "/"
            for d in path:
                location += "/" + d
        return all_vals + [location]

    def get_upgrade_costs(self):
        """
        Returns the total mana cost to fully upgrade this item.
        
        Returns:
            float: The cost
        """
        if self.remaining_upgrades == 0:
            return 0
        if self.type == "Accessory":
            return ACC_UPGRADE_COST
        return sum((UPGRADE_COST_BASE * level ** UPGRADE_COST_EXP for level in range(self.level,
                                                                                     self.max_upgrades)))

    def get_weighted_score(self, weights, upgrade_accs=True):
        """
        Simulates upgrading the item to max rank, stats with higher weights have priority
        Maximizes resistances for non builders

        Parameters:
            weights (dict{string : float}): Maps stat types to weights
            upgrade_accs (bool): If False: Only use current stats for accessories

        Returns:
            int: Weighted sum of all stats
        """
        max_resistance = False
        # optimizing for hero stats means resistances are important
        if max(weights[:4]) > 0 and self.type == "Armor":
            max_resistance = True
        if len(weights) != len(self.stat_dict) - 4:
            raise Exception(f"Length of given weights ({len(weights)}) " +
                            f"must match number of stats ({len(self.stat_dict) - 4})")
        remaining_upgrades = self.remaining_upgrades
        if max_resistance:
            # item useless for non builder
            if self.remaining_upgrades_res < 0:
                return 0, list(self.stat_dict.values())[4:]
            remaining_upgrades = self.remaining_upgrades_res
        effective_stats = self.get_post_upgrade_stats(remaining_upgrades, copy.deepcopy(weights), upgrade_accs)
        if self.type == "Armor":
            quality_mult = QUALITY_PROPERTIES[self.quality][0]
            boosted_stats = effective_stats > 0
            boosted_stats = boosted_stats * quality_mult + ~boosted_stats
            effective_stats = effective_stats * boosted_stats
        score = np.sum(effective_stats * np.array(weights))
        if sum(weights) == 0:
            return score
        return (score / sum(weights)), [math.ceil(x) for x in effective_stats]

    def get_post_upgrade_stats(self, remaining_upgrades, weights, upgrade_accs):
        """
        Calculates equipment stats at maximum level by distributing all remaining levels according to
        given weights.

        Args:
            remaining_upgrades (int): Number of available upgrades
            weights (dict{string : float}): Maps stat types to weights
            upgrade_accs (bool): If False: Do not upgrade accessories

        Returns:
            list(int): Contains all post upgrade stats, excluding resistances
        """
        effective_stats = list(self.stat_dict.values())[4:]
        if self.type == "Accessory" and not upgrade_accs:
            return np.array(effective_stats)
        while remaining_upgrades > 0 and max(weights) > 0:
            best_stat = np.argmax(np.array(weights))
            if effective_stats[best_stat] == 0:
                weights[best_stat] = -float_info.max
                continue
            quality_cap = QUALITY_PROPERTIES[self.quality][1]
            if self.type == "Familiar" and self.quality == 0:
                if "Diamond" in self.desc_string:
                    quality_cap = 800
                elif "Treadmill_on_itself" in self.desc_string:
                    quality_cap = 700
            free_upgrades = max(quality_cap - effective_stats[best_stat], 0)
            actual_upgrades = min(remaining_upgrades, free_upgrades)
            effective_stats[best_stat] += actual_upgrades
            remaining_upgrades -= actual_upgrades
            weights[best_stat] = -1
        return np.array(effective_stats)

    def __str__(self):
        """
        Creates PrettyTable and adds equipments stats as a row

        Returns:
            PrettyTable: Contains stats
        """
        header = ["Type", "Pos", "Material", "Tier"] + list(OFFSET_DICT.keys())[4:] + \
                 ["Level"]
        p_table = PrettyTable(header)
        p_table.add_row(self.get_property_list())
        return p_table.__str__()
