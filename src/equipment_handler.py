import copy

import numpy as np
import multiprocessing
import os
import math
from prettytable import PrettyTable
from colorama import Fore, Back, Style
from src.consts import *
from src.data_handler import DataHandler
from src.equipment import Equipment
from itertools import product as cart_product


class EquipmentHandler:
    """
    Holds all equipment in one save file and provides methods for equipment optimization
    """

    def __init__(self, num_threads=-1, raw_output=False, path=DECOMP_FILE_PATH, armor_only=False):
        """
        Constructor for EquipmentHandler class. Reads stats for every equipment
        instance from save file in parallel

        Parameters:
            num_threads (int): Number of threads to use, -1 = use all available
            raw_output (bool): Disables ANSI escape characters
            path (string): Save file location
            armor_only (bool): Ignore non armor equipment
        """
        self.data_handler = DataHandler(path, compressed=False)
        self.last_item_index = self.data_handler.last_equipment_index
        self.dirs = self.data_handler.get_dir_ids()
        self.all_equipment = []
        self.num_threads = num_threads
        self.raw_output = raw_output
        self.armor_only = armor_only
        self.target_classes = self.data_handler.get_all_target_classes()
        self.print_targets = list(self.target_classes.keys())
        self.print_type_filter = list(EQUIPMENT_TYPES.keys())
        self.print_csv = False
        if self.num_threads == -1:
            self.num_threads = os.cpu_count()
        worker_pool = multiprocessing.Pool(processes=self.num_threads)
        part_length = int(math.ceil(self.last_item_index / self.num_threads))
        indices_list = []
        for i in range(self.num_threads):
            first_index = i * part_length
            last_index = min(self.last_item_index, (i + 1) * part_length)
            indices_list.append((first_index, last_index))
        worker_output = worker_pool.map(self.init_worker, indices_list)
        # remove duplicates
        found_equip = [e for sl in worker_output for e in sl]
        for equip in found_equip:
            if equip.index not in [e.index for e in self.all_equipment]:
                self.all_equipment.append(equip)
        self.reserved_equipment = [False] * len(self.all_equipment)

    def init_worker(self, indices):
        """
        Function to run in worker threads to initialize equipment

        Parameters:
            indices (list[int, int]): Start and end index in save file to check for equipment

        Returns:
            list[Equipment]: The equipment found in the save file
        """
        found_equip = []
        start_index, end_index = indices
        cur_index, found = self.data_handler.find_next_string_index(
            EQUIPMENT_MARKER, start_index=start_index, raw=True, limit=self.last_item_index)
        on_floor = False
        while found and cur_index <= end_index:
            cur_equip, cur_index = self.data_handler.get_all_equip_stats(cur_index + 5)
            cur_index, found = self.data_handler.find_next_string_index(
                EQUIPMENT_MARKER, start_index=cur_index, raw=True, limit=self.last_item_index - cur_index)
            if cur_equip is not None:
                # some event items are bugged and show way too many open upgrades
                if cur_equip.level > 700:
                    cur_equip.level = cur_equip.max_upgrades
                if cur_equip.owner == "Tavern Floor":
                    on_floor = True
                if on_floor and cur_equip.dir_id == 2 ** 32 - 1 and \
                        cur_equip.pos_x == 0 and cur_equip.pos_y == 0 and cur_equip.pos_z == 0:
                    cur_equip.owner = "Shop"
                    cur_equip.dir_id = -1
                found_equip.append(cur_equip)
        return found_equip

    def optimize_for_targets(self, targets, print_all=False, upgrade_accs=True):
        """
        Finds a set of equipment for each given target by optimizing for given weights.
        Targets have descending priority for equipment by their order in given targets dict.
        Assigns one item per slot of target character if target is a builder. Ignores weapons and
        pets for non builders.

        Parameters:
            targets (dict{string: dict{string, float}}): Maps target names to weights for optimization
            print_all (bool): Print best equipment set for each armor material instead only the top one
            upgrade_accs (bool): If False: Only use current stats for accessories
        """
        upgrade_cost = 0
        upgrade_cost_no_accs = 0
        for target in targets.keys():
            weights = self.extract_weights(target, targets)
            all_equips, all_stats, all_scores = self.optimize_by_weights(weights, target, upgrade_accs=upgrade_accs)
            if target in self.print_targets:
                self.print_optimization_results(all_equips, all_stats, all_scores, weights, target,
                                                print_all, upgrade_accs)
            for e in all_equips[-1]:
                self.reserved_equipment[e] = True
                if target in self.print_targets:
                    cost = self.all_equipment[e].get_upgrade_costs()
                    upgrade_cost += cost
                    upgrade_cost_no_accs += cost if self.all_equipment[e].type != "Accessory" else 0
        print(f"Estimated upgrade cost for all characters (without accessories): " +
              f"{np.round(upgrade_cost / 1e9, 2)}B" +
              f" ({np.round(upgrade_cost_no_accs / 1e9, 2)}B)")

    def extract_weights(self, target, targets):
        """
        Parameters:
            target (string): Name of the target for which weights are to be read
            targets (dict{string: dict{string, float}}): Maps target names to weights for optimization

        Returns:
            (dict{string: float}): Maps stat name to weight
        """
        if target not in self.target_classes.keys():
            raise Exception(f"Unknown optimization target '{target}'")
        for key in targets[target].keys():
            if key not in STAT_OFFSET_DICT.keys():
                raise Exception(f"Unknown stat '{key}'")
        weights = np.zeros(len(STAT_OFFSET_DICT.keys()) - 4)
        for i, stat in enumerate(list(STAT_OFFSET_DICT.keys())[4:]):
            if stat in targets[target]:
                weights[i] = targets[target][stat]
        total = sum(weights)
        return weights if total == 0 else weights / total

    def optimize_by_weights(self, weights, target, cannot_steal=False, protected=None,
                            upgrade_accs=True):
        """
        Finds optimal equipment by maximizing a weighted score of all stats.
        Only considers possible sets of the same material for armor. Prints optimization
        results to stdout.

        Parameters:
            weights (dict{string: float}): Maps stat name to weight
            target (string): Target character for optimization
            print_all (bool): Print best equipment set for each armor material instead only the top one
            cannot_steal (bool): Ignore items already equipped on another character
            protected (list[string]): Characters whose items will not be reassigned
            upgrade_accs (bool): If False: Only use current stats for accessories

        Returns:
            list[int]: Indices of equipment found during optimization
            float: Optimization score
        """
        if protected is None:
            protected = []
        all_equips = []
        all_stats = []
        all_scores = []
        # armors
        for equip_mat in ARMOR_MATERIALS:
            all_equips, all_stats, all_scores = self.optimize_for_slots(
                ARMOR_SLOTS, weights, target, cannot_steal, protected,
                all_equips, all_stats, all_scores, equip_mat, upgrade_accs)
        all_equips = [e for _, e in sorted(zip(all_scores, all_equips))]
        all_stats = [e for _, e in sorted(zip(all_scores, all_stats))]
        all_scores = sorted(all_scores)
        if not self.armor_only:
            # accessories
            acc_slots = list(ACCESSORY_SLOTS.keys())[:-1]
            if self.target_classes[target] == "Squire":
                acc_slots.append("Shield")
            all_equips, all_stats, all_scores = self.optimize_for_slots(
                acc_slots, weights, target, cannot_steal, protected,
                all_equips, all_stats, all_scores, upgrade_accs=upgrade_accs)
            if max(weights[:4]) <= 0:
                # pets and weapons
                all_equips, all_stats, all_scores = self.optimize_for_slots(
                    CHAR_SLOTS[self.target_classes[target]], weights, target, cannot_steal, protected,
                    all_equips, all_stats, all_scores, upgrade_accs=upgrade_accs)
            else:
                for i, equip in enumerate(self.all_equipment):
                    if equip.owner == target and equip.type in ["Weapon", "Familiar"]:
                        for j in range(len(all_equips)):
                            all_equips[j].append(i)
                            cur_score, cur_stats = equip.get_weighted_score(weights, upgrade_accs)
                            all_stats[j] += cur_stats
                            all_scores[j] += cur_score
        return all_equips, all_stats, all_scores

    def optimize_for_slots(self, slots, weights, target, cannot_steal, protected,
                           prev_equips, prev_stats, prev_scores, equip_mat=None, upgrade_accs=True):
        """
        Finds optimal equipment for every given slot and weight.

        Parameters:
            slots (list[string]): Slots to find equipment for
            weights (dict{string: float}): Maps stat name to weight
            target (string): Target character for optimization
            cannot_steal (bool): Ignore items already equipped on another character
            protected (list[string]): Characters whose items will not be reassigned
            equip_mat (string): Material of the armor set, e.g. Chain
            prev_equips (list[list[int]]): Indices of equipment from previous optimization
            prev_stats (list[list[float]]): Stats from previous optimization
            prev_scores (list[float]): Scores from previous optimization
            upgrade_accs (bool): If False: Only use current stats for accessories

        Returns:
            list[list[int]]: prev_equips with newly found Equipment appended
            list[list[float]]: prev_stats with new stats appended
            list[float]: prev_scores with new score appended
        """
        best_equips = []
        equip_stats = [0] * (len(STAT_OFFSET_DICT) - 4)
        equip_score = 0
        for target_slots in slots:
            best_score = 0
            best_index = -1
            best_stats = 0
            for i, equipment in enumerate(self.all_equipment):
                if equipment.slot not in target_slots or equipment.material != equip_mat or \
                        equipment.owner != "" and (equipment.owner != target) and cannot_steal or \
                        equipment.owner in protected or self.reserved_equipment[i]:
                    continue
                cur_score, cur_stats = equipment.get_weighted_score(weights, upgrade_accs)
                if cur_score > best_score and i not in best_equips:
                    best_score = cur_score
                    best_index = i
                    best_stats = cur_stats
            if best_index != -1:
                best_equips.append(best_index)
                equip_stats = list(np.array(best_stats) + np.array(equip_stats))
                equip_score += best_score
        if equip_mat is not None:
            prev_equips.append(best_equips)
            prev_scores.append(equip_score)
            prev_stats.append(equip_stats)
        else:
            for i in range(len(prev_equips)):
                prev_equips[i] += best_equips
                prev_scores[i] += equip_score
                prev_stats[i] = [eq_s + prev_s for eq_s, prev_s in zip(equip_stats, prev_stats[i])]
        return prev_equips, prev_stats, prev_scores

    def find_obsolete_equipment(self):
        """
        Finds all equipment that is pareto dominated by at least one other equipment of the same type, slot
        and material. This ignores resistances and all damage related values on weapons and pets.
        """
        obsolete_equip = []
        for i, first_equip in enumerate(self.all_equipment):
            for j, second_equip in enumerate(self.all_equipment):
                if j != i and first_equip.type == second_equip.type \
                        and first_equip.slot == second_equip.slot \
                        and first_equip.material == second_equip.material \
                        and second_equip.pareto_dominates(first_equip):
                    obsolete_equip.append(i)
                    break
        if len(obsolete_equip) == 0:
            print("No obsolete equipment found!")
        else:
            print(f"{self.generate_table(obsolete_equip)}\nTotals:\n{self.generate_counts_table(obsolete_equip)}")

    def optimize_with_conditions(self, targets, conditions=None, print_all=False,
                                 upgrade_accs=True, add_speed_condition=True):
        if conditions is None:
            conditions = {}
        for target in conditions:
            if target not in self.target_classes:
                raise Exception(f"Unknown target in conditions '{target}'")
        reserved_equip = []
        for target in targets:
            weights = self.extract_weights(target, targets)
            target_conditions = []
            if target in conditions:
                target_conditions = conditions[target]
            if add_speed_condition:
                target_conditions.append(["HSPD", lambda a, b: a > b, 99])
            reserved_equip = np.append(reserved_equip, self.optimize_for_target(
                target, weights, target_conditions, reserved_equip, print_all, upgrade_accs))

    def optimize_for_target(self, target, weights, conditions, reserved_equip,
                            print_all=False, upgrade_accs=True):
        # map equip type to list of (equip_id, score) tuples
        slot_dict = {}
        weapon_slots = set()
        for slots in CHAR_SLOTS[self.target_classes[target]]:
            for slot in slots:
                if slot != "Pet":
                    weapon_slots.add(slot)
        for i, equip in enumerate(self.all_equipment):
            if i in reserved_equip:
                continue
            if equip.type == "Weapon":
                if equip.slot not in weapon_slots:
                    continue
                slot = "Weapon"
            elif equip.type == "Armor":
                slot = equip.slot + equip.material
            elif equip.type == "Accessory":
                slot = equip.slot
            else:
                slot = "Pet"
            score, stats = equip.get_weighted_score(weights, upgrade_accs)
            if slot not in slot_dict.keys():
                slot_dict[slot] = [(i, score, stats)]
            else:
                slot_dict[slot].append((i, score, stats))
        # get all slots relevant to this target
        target_slots = ["Hat", "Mask", "Bracers"]
        if self.target_classes[target] == "Squire":
            target_slots.append("Shield")
        if max(weights[:4]) <= 0:
            for slots in CHAR_SLOTS[self.target_classes[target]]:
                if slots[0] == "Pet":
                    target_slots.append("Pet")
                else:
                    target_slots.append("Weapon")
        # remove pareto dominated items
        for slot in slot_dict.keys():
            num_iterations = 1
            if slot in ["Weapon", "Pet"]:
                num_iterations = target_slots.count(slot)
            obsoletes = copy.deepcopy(slot_dict[slot])
            for i in range(num_iterations):
                cur_obsoletes = []
                for ((id_0, score_0, _), (id_1, score_1, stats_1)) in cart_product(obsoletes, obsoletes):
                    if id_0 == id_1:
                        continue
                    if score_0 < score_1:
                        continue
                    is_dominating = True
                    for stat, cond_fun, _ in conditions:
                        if not cond_fun(self.all_equipment[id_0].stat_dict[stat],
                                        self.all_equipment[id_1].stat_dict[stat]):
                            is_dominating = False
                            break
                    if is_dominating:
                        cur_obsoletes.append((id_1, score_1, stats_1))
                obsoletes = copy.deepcopy(cur_obsoletes)
            slot_dict[slot] = [(e_id, score, stats) for e_id, score, stats in slot_dict[slot]
                               if e_id not in [i for i, _, _ in obsoletes]]
        # get pet and weapon stats for non builders
        own_equip = []
        own_stats = np.zeros_like(weights, dtype=np.int32)
        own_score = 0
        if max(weights[:4]) > 0:
            for i, equip in enumerate(self.all_equipment):
                if equip.owner == target and equip.type in ["Weapon", "Familiar"]:
                    score, stats = equip.get_weighted_score(weights, upgrade_accs)
                    own_score += score
                    own_stats += stats
                    own_equip.append(i)
        all_scores = []
        all_equips = []
        all_stats = []
        for material in ARMOR_MATERIALS:
            best_score = -1
            best_equip = []
            best_stats = []
            cur_slots = copy.copy(target_slots)
            for pair in cart_product(ARMOR_SLOTS, [material]):
                cur_slots.insert(0, "".join(pair))
            #print([f"{s}: {len(slot_dict[s])}" for s in slot_dict if s in cur_slots])
            #print(np.prod([len(slot_dict[s]) for s in slot_dict if s in cur_slots]))
            # iterate over cartesian product of equip for slots of target
            for loadout in cart_product(*[slot_dict[slot] for slot in cur_slots]):
                cur_equip = [equip for equip, score, stats in loadout] + own_equip
                if len(np.unique(cur_equip)) != len(cur_equip):
                    continue
                is_valid = True
                # find the best combination that fulfills all conditions
                for stat, cond_fun, threshold in conditions:
                    stat_total = sum([self.all_equipment[e].stat_dict[stat] for e in cur_equip])
                    if not cond_fun(stat_total, threshold):
                        is_valid = False
                        break
                if not is_valid:
                    continue
                cur_score = sum([score for equip, score, stats in loadout]) + own_score
                if cur_score > best_score:
                    best_score = cur_score + own_score
                    best_equip = cur_equip
                    best_stats = np.sum([stats for equip, score, stats in loadout], axis=0) + own_stats
            all_equips.append(best_equip)
            all_scores.append(best_score)
            all_stats.append(best_stats)
        indices = np.argsort(all_scores)
        all_equips = np.array(all_equips)[indices]
        all_stats = np.array(all_stats)[indices]
        all_scores = np.array(all_scores)[indices]
        self.print_optimization_results(all_equips, all_stats, all_scores, weights, target,
                                        print_all, upgrade_accs)
        return all_equips[-1]

    def print_optimization_results(self, all_equips, all_stats, all_scores, weights, target,
                                   print_all=False, upgrade_accs=True):
        """
        Prints equipment found during optimization

        Parameters:
            all_equips (list[list[int]]): Contains indices of equipment found during optimization runs
            all_stats (list[list[float]]): Stats of given equipment
            all_scores (list[float]): Scores of given equipment sets
            weights (dict{string: float}): Maps stat name to weight
            target (string): Target character for optimization
            print_all (bool): Print best equipment set for each armor material instead only the top one
            upgrade_accs (bool): If False: Only use current stats for accessories
        """

        def pos_neg_color(num):
            color = Fore.YELLOW
            if num < 0:
                color = Fore.RED
            elif num > 0:
                color = Fore.GREEN
            return color

        delim = "=" * 60
        header = list(STAT_OFFSET_DICT.keys())[4:]
        start_index = 0
        is_dps_target = True if max(weights[:4]) > 0 else False
        if not print_all:
            start_index = len(all_scores) - 1
        found_equip = False
        for i in range(start_index, len(all_scores)):
            equip, stats, score = all_equips[i], all_stats[i], all_scores[i]
            if len(equip) == 0:
                continue
            found_equip = True
            score = math.ceil(score)
            target_str = target
            if not self.raw_output:
                target_str = Fore.BLUE + Style.BRIGHT + target + Style.RESET_ALL
            print(f"Equipment for {target_str}:")
            print(delim)
            print(f"Rank {len(all_scores) - i} ({self.all_equipment[equip[0]].material})" + ":\n" +
                  self.generate_table(equip).__str__())
            p_table = PrettyTable(header)
            stats = list(stats)
            if not self.raw_output:
                for j, s in enumerate(stats):
                    if weights[j] > 0:
                        stats[j] = Style.BRIGHT + str(s) + Style.RESET_ALL
                    else:
                        stats[j] = Style.DIM + str(s) + Style.RESET_ALL
            p_table.add_row(list(stats))
            print(f"Stats with score {score}:\n" + p_table.__str__())
            owner_stats = np.zeros(len(weights))
            owner_score = 0
            for e in self.all_equipment:
                if e.owner == target and not \
                        (self.armor_only and e.type != "Armor"):
                    item_score, item_stats = e.get_weighted_score(weights, upgrade_accs)
                    owner_stats += np.array(item_stats)
                    owner_score += item_score
            owner_score = math.ceil(owner_score)
            stat_diffs = [int(e) for e in list(np.array(all_stats[-1]) - owner_stats)]
            if not self.raw_output:
                for j in range(len(stat_diffs)):
                    if weights[j] > 0:
                        stat_diffs[j] = pos_neg_color(stat_diffs[j]) + str(stat_diffs[j]) + \
                                        Style.RESET_ALL
            p_table = PrettyTable(header)
            p_table.add_row(stat_diffs)
            print(f"Stat changes with score delta " +
                  (pos_neg_color(score - owner_score) if not self.raw_output else '') +
                  str(score - owner_score) +
                  (Style.RESET_ALL if not self.raw_output else '') + ":\n" +
                  p_table.__str__())
            upgrade_cost = sum([self.all_equipment[e].get_upgrade_costs() for e in equip])
            upgrade_cost_no_accs = sum([self.all_equipment[e].get_upgrade_costs()
                                        if self.all_equipment[e].type != "Accessory" else 0 for e in equip])
            print(f"Estimated upgrade cost (without accessories): {np.round(upgrade_cost / 1e9, 2)}B" +
                  f" ({np.round(upgrade_cost_no_accs / 1e9, 2)}B)")
        if not found_equip:
            print(Fore.RED + f"No equipment found for {Style.BRIGHT}{target}!" + Style.RESET_ALL)
        print(delim)

    def generate_table(self, indices, sort=False):
        """
        Generates PrettyTable for equipment with given indices

        Parameters:
            sort (bool): If true: Sort equipment by its index
            indices (list[int]): Indices for equipment

        Returns:
            PrettyTable: Contains stats for given equipment
        """
        header = ["Type", "Slot", "Material", "Tier"] + \
            list(STAT_OFFSET_DICT.keys())[0 if self.print_csv else 4:] + \
            ["Level", "Location"]
        p_table = PrettyTable(header)
        if sort:
            all_equip = sorted(self.all_equipment, key=lambda e: e.index)
        else:
            all_equip = self.all_equipment
        for i in indices:
            cur_equip = all_equip[i]
            if cur_equip.type in self.print_type_filter:
                p_table.add_row(cur_equip.get_property_list(self.dirs, self.print_csv))
        if self.print_csv:
            return p_table
        if self.raw_output:
            return p_table.__str__()
        table = ""
        for i, row in enumerate(p_table.__str__().split("\n")):
            if i > 2 and i % 2 == 0 and row[0] != "+":
                row = row[0] + Back.LIGHTBLACK_EX + row[1:-1] + Style.RESET_ALL + row[-1]
            table += row + "\n"
        table = table[:-1]
        return table

    def generate_counts_table(self, indices=None):
        """
        Generates a PrettyTable containing the number of equipment pieces for each type

        Parameters:
            indices (list[int]): Indices of equipment included in the count. If this is None,
            all equipment will be used

        Returns:
            PrettyTable: The generated table
        """
        if indices is None:
            indices = np.arange(len(self.all_equipment))
        type_counts = {}
        for equip_type in EQUIPMENT_TYPES.keys():
            type_counts[equip_type] = 0
        for i in indices:
            equip = self.all_equipment[i]
            for equip_type in EQUIPMENT_TYPES.keys():
                if equip.type == equip_type:
                    type_counts[equip.type] += 1
                    break
        p_table = PrettyTable(type_counts.keys())
        p_table.add_row(type_counts.values())
        return p_table

    def __str__(self):
        """
        Generates PrettyTable containing every equipment in self.all_equipment

        Returns:
            string: String form of generated PrettyTable
        """
        equip_table = self.generate_table(list(range(len(self.all_equipment))), sort=True)
        if self.print_csv:
            return equip_table.get_csv_string()
        counts_table = self.generate_counts_table()
        return equip_table.__str__() + "\nTotals:\n" + counts_table.__str__()
