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
        self.is_equipped = False
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
        self.quality_cap = 0
        self.quality_mult = 0

    def init_upgrades(self):
        """
        Calculates available upgrades, assumes non builders always max resistances on armor.
        Must be called after initializing max_upgrades and level
        """
        if self.type == "Accessory":
            self.quality_cap = 500 if self.quality > 14 else 360
        elif self.type == "Familiar" and self.quality == 0:
            if "Diamond" in self.desc_string:
                self.quality_cap = 800
            elif "Treadmill_on_itself" in self.desc_string:
                self.quality_cap = 700
        else:
            self.quality_cap = QUALITY_PROPERTIES[self.quality][1]
        self.quality_mult = QUALITY_PROPERTIES[self.quality][0] if self.type == "Armor" else 1
        self.remaining_upgrades = self.max_upgrades - self.level
        self.remaining_upgrades_res = self.remaining_upgrades
        resistances = list(self.stat_dict.values())[:4]
        res_target = 40 / QUALITY_PROPERTIES[self.quality][0]
        cur_level = self.level
        soft_cap = 23
        # assume pre upgraded equipped items are good
        if min(resistances) > 20 and self.level > 100 and self.is_equipped:
            return
        # simulate soft-capping resistances
        for i in range(len(resistances)):
            # missing resistance -> useless
            if resistances[i] == 0:
                self.remaining_upgrades_res = -1
                return
            while resistances[i] < soft_cap:
                if cur_level % 10 == 9:
                    found = False
                    for j in range(len(resistances)):
                        if soft_cap <= resistances[j] < res_target:
                            resistances[j] += 1
                            cur_level += 1
                            found = True
                            break
                    if found:
                        continue
                if resistances[i] == 20 or resistances[i] == 21:
                    res_offset = 3
                elif 14 <= abs(resistances[i]) < 20 or resistances[i] == -1:
                    res_offset = 2
                else:
                    res_offset = 1
                resistances[i] += res_offset
                cur_level += 1
                # cannot max resistances -> useless
                if self.remaining_upgrades_res == 0 and min(resistances) < res_target:
                    self.remaining_upgrades_res = -1
                    return
        self.remaining_upgrades_res -= cur_level - self.level
        # simulate capping resistances
        res_target = math.ceil(40 / QUALITY_PROPERTIES[self.quality][0])
        res_upgrades_needed = sum([res_target - res for res in resistances])
        res_upgrades_available = math.floor(self.remaining_upgrades_res / 10) \
            + math.floor((cur_level % 10 + self.remaining_upgrades_res % 10) / 10)
        if res_upgrades_needed > res_upgrades_available:
            self.remaining_upgrades_res = -1
            return
        else:
            self.remaining_upgrades_res -= res_upgrades_needed

    def get_property_list(self, dirs=None, include_res=False):
        """
        Returns a list of item stats and properties

        Parameters:
            dirs (dict{int: list[string, int]}) Maps directory id to its name and parent id
            include_res (bool): If True: includes resistance values

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

    def get_boosted_stats(self, effective_stats):
        """
        Calculates effective stats after armor set boni.

        Parameters:
            effective_stats (np.array): Contains current stats of this item

        Returns:
            np.array: The boosted stats
        """
        boosted_stats = np.array(copy.deepcopy(effective_stats))
        if self.type != "Armor":
            return boosted_stats
        boosted_stats_mask = boosted_stats > 0
        boosted_stats_mask = boosted_stats_mask * self.quality_mult + ~boosted_stats_mask
        boosted_stats = boosted_stats * boosted_stats_mask
        return np.ceil(boosted_stats).astype(int)

    def get_weighted_score(self, weights, upgrade_accs=True, condition_stats=None):
        """
        Simulates upgrading the item to max rank, stats with higher weights have priority
        Maximizes resistances for non builders

        Parameters:
            condition_stats (list[String]): Contains names of stats which are relevant to optimization
                conditions
            weights (dict{string : float}): Maps stat types to weights
            upgrade_accs (bool): If False: Only use current stats for accessories

        Returns:
            If condition_stats is None:
                int: Weighted sum of all stats,
                np.array: Stats after upgrading
            Else:
                int: Weighted sum of all stats,
                np.array: Stats after upgrading,
                dict{string: list[int, string, int, float]}: Maps stats which are relevant to a condition
                    to the weight, source stat, number and quality multiplier of upgrades which could be applied
                    to that stat
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
                if condition_stats is not None:
                    return 0, self.get_boosted_stats(list(self.stat_dict.values())[4:]), {}
                return 0, self.get_boosted_stats(list(self.stat_dict.values())[4:])
            remaining_upgrades = self.remaining_upgrades_res
        effective_stats, potential_upgrades = self.get_post_upgrade_stats(
            remaining_upgrades, copy.deepcopy(weights), upgrade_accs, condition_stats)
        effective_stats = self.get_boosted_stats(effective_stats)
        score = np.sum(effective_stats * np.array(weights))
        if condition_stats is not None:
            return score, effective_stats, potential_upgrades
        return score, effective_stats

    def get_post_upgrade_stats(self, remaining_upgrades, weights, upgrade_accs, condition_stats=None):
        """
        Calculates equipment stats at maximum level by distributing all remaining levels according to
        given weights. Also determines how many upgrades could be applied to each condition stat.

        Parameters:
            condition_stats (list[String]): Contains names of stats which are relevant to optimization
                conditions
            remaining_upgrades (int): Number of available upgrades
            weights (dict{string : float}): Maps stat types to weights
            upgrade_accs (bool): If False: Do not upgrade accessories

        Returns:
            list(int): Contains all post upgrade stats, excluding resistances,
            dict{string: list[int, string, int, float]}: Maps stats which are relevant to a condition
                to the weight, source stat, number and quality multiplier of upgrades which could be applied
                to that stat
        """
        potential_upgrades = {}
        effective_stats = list(self.stat_dict.values())[4:]
        if self.type == "Accessory" and not upgrade_accs:
            return np.array(effective_stats), potential_upgrades
        while remaining_upgrades > 0 and max(weights) > 0:
            best_stat = np.argmax(np.array(weights))
            if effective_stats[best_stat] == 0:
                weights[best_stat] = -float_info.max
                continue
            free_upgrades = max(self.quality_cap - effective_stats[best_stat], 0)
            actual_upgrades = min(remaining_upgrades, free_upgrades)
            effective_stats[best_stat] += actual_upgrades
            remaining_upgrades -= actual_upgrades
            # track all stat points which could be spent to fulfill conditions
            if condition_stats is not None:
                stat_name = list(STAT_OFFSET_DICT.keys())[best_stat + 4]
                for target_stat in condition_stats:
                    if self.stat_dict[target_stat] == 0:
                        continue
                    new_item = [weights[best_stat], stat_name,
                                min(actual_upgrades, self.quality_cap - self.stat_dict[target_stat]),
                                self.quality_mult]
                    if target_stat not in potential_upgrades:
                        potential_upgrades[target_stat] = [new_item]
                    else:
                        potential_upgrades[target_stat].append(new_item)
            weights[best_stat] = -float_info.max

        if remaining_upgrades > 0 and condition_stats is not None:
            for target_stat in condition_stats:
                if self.stat_dict[target_stat] == 0:
                    continue
                new_item = [0, "", min(remaining_upgrades, self.quality_cap - self.stat_dict[target_stat]),
                            self.quality_mult]
                if target_stat not in potential_upgrades:
                    potential_upgrades[target_stat] = [new_item]
                else:
                    potential_upgrades[target_stat].append(new_item)
        return np.array(effective_stats), potential_upgrades

    def pareto_dominates(self, other):
        """
        Checks if this equipment pareto dominates given other equipment. This ignores
        all damage related values on weapons and pets.

        Parameters:
            other (equipment): Equipment for comparison

        Returns:
            bool: True if and only if this pareto dominates other
        """
        def prepare_stats(equip):
            stats = np.array(list(equip.stat_dict.values())[:4])
            stats -= (stats == 0) * 100
            return np.concatenate((stats, np.array(list(equip.stat_dict.values())[4:]),
                                   np.array([equip.remaining_upgrades])))

        own_stats = prepare_stats(self)
        other_stats = prepare_stats(other)
        return np.sum(own_stats >= other_stats) == len(own_stats)

    def __str__(self):
        """
        Creates PrettyTable and adds equipments stats as a row

        Returns:
            PrettyTable: Contains stats
        """
        header = ["Type", "Pos", "Material", "Tier"] + list(STAT_OFFSET_DICT.keys())[4:] + \
                 ["Level"]
        p_table = PrettyTable(header)
        p_table.add_row(self.get_property_list())
        return p_table.__str__()
