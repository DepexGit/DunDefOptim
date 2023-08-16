import copy
import heapq
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
from itertools import islice
from multiprocessing import Pool
from functools import partial


def is_non_builder(weights):
    """
    Checks if weights belong to a non builder, i.e. if weights for hero stats are non-zero

    Parameters:
        weights (list[float]): Contains weights for every stat

    Returns:
        bool: True if non builder, else False
    """
    return max(weights[:4]) > 0


def equipment_is_valid(conditions, stats):
    """
    Checks if equipment fulfills all conditions.

    Parameters:
        conditions (list[list[string, condition class, int]]): Contains condition stats, typer and thresholds
        stats (list[int]): Contains stats for equipment

    Returns:
        bool: True if equipment fulfills conditions, else False
    """
    for stat_name, cond_fun, threshold in conditions:
        stat_value = stats[STAT_INDEX_DICT[stat_name]]
        if not cond_fun.apply(stat_value, threshold):
            return False
    return True


def find_best_combination(combinations, conditions, own_equip, own_score, own_stats):
    """
    Iterates over all given combinations and finds the one with the highest score which also fulfills
    all given conditions.

    Parameters:
        combinations (iterable(tuple(list[int], float, list[int],
                      dict{string: list[float, string, int, float]}))):
            Contains a list of equipment, their scores, their stats and a dict containing potential upgrades
        conditions (list[list[string, condition class, int]]): Contains condition stats, typer and thresholds
        own_equip (list[int]): Ids of equipment belonging to optimization target
        own_score (float): Score of equipment belonging to optimization target
        own_stats (list[int]): Stats of equipment belonging to optimization target

    Returns:
        list[int]: Ids of the best equipment,
        float: Score of the best equipment,
        list[int]: Stats of the best equipment
    """
    best_score = -1
    best_equip = []
    best_stats = []
    for loadout in combinations:
        cur_equip = [equip for equip, score, stats, _ in loadout] + own_equip
        if len(np.unique(cur_equip)) != len(cur_equip):
            continue
        # find the best combination that fulfills all conditions
        cur_score = sum([score for equip, score, stats, _ in loadout]) + own_score
        if cur_score <= best_score:
            continue
        cur_stats = np.sum([stats for equip, score, stats, _ in loadout], axis=0) + own_stats
        is_valid = equipment_is_valid(conditions, cur_stats)
        if not is_valid:
            missing_stats = {}
            fails_at_most = False
            for stat_name, cond_fun, threshold in conditions:
                stat_value = cur_stats[STAT_INDEX_DICT[stat_name]]
                if not cond_fun.apply(stat_value, threshold):
                    if cond_fun.name == "at_most":
                        fails_at_most = True
                        break
                    missing_stats[stat_name] = threshold - stat_value
            if fails_at_most:
                continue
            upgrade_queue = []
            for equip, potential_upgrades in zip(cur_equip, [pot_ups for equip, score, stats, pot_ups in loadout]):
                for key, values in potential_upgrades.items():
                    for value in values:
                        heapq.heappush(upgrade_queue, value + [key] + [equip])
            while len(upgrade_queue) > 0 and sum(missing_stats.values()) > 0 and cur_score > best_score:
                weight, source_stat, upgrades, mult, target_stat, equip = heapq.heappop(upgrade_queue)
                if upgrades == 0 or target_stat not in missing_stats:
                    continue
                used_upgrades = min(int(np.ceil(missing_stats[target_stat] / mult)), upgrades)
                stat_delta = int(np.ceil(used_upgrades * mult))
                missing_stats[target_stat] -= stat_delta
                cur_stats[STAT_INDEX_DICT[target_stat]] += stat_delta
                if source_stat != "":
                    cur_stats[STAT_INDEX_DICT[source_stat]] -= stat_delta
                    cur_score -= weight * stat_delta
                for i in range(len(upgrade_queue)):
                    if upgrade_queue[i][0] == weight and upgrade_queue[i][4] == equip:
                        upgrade_queue[i] = [weight, source_stat,
                                            upgrades - used_upgrades, mult, target_stat, equip]
                if sum(missing_stats.values()) <= 0:
                    is_valid = True
                    break
        if cur_score > best_score and is_valid:
            best_score = cur_score
            best_equip = cur_equip
            best_stats = cur_stats
    return best_equip, best_score, best_stats


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

    def optimize_for_targets(self, targets, conditions=None, print_all=False, upgrade_accs=True,
                             add_speed_condition=True, protected=None):
        """
        Finds a set of equipment for each given target by optimizing for given weights.
        Targets have descending priority for equipment by their order in given targets dict.
        Assigns one item per slot of target character if target is a builder. Ignores weapons and
        pets for non builders. Ensures output equipment satisfies all given conditions.

        Parameters:
            targets (dict{string: dict{string, float}}): Maps target names to weights for optimization
            print_all (bool): Print best equipment set for each armor material instead only the top one
            upgrade_accs (bool): If False: Only use current stats for accessories
            conditions (dict{string: list[list[string, condition class, int]]}):
                Maps target names to list of conditions
            add_speed_condition: If True: Ensure all equipment will grant at least 100 movement speed
            protected (list[string]): Targets to be ignored during optimization
        """
        if protected is None:
            protected = {}
        for target in protected:
            for i, equip in enumerate(self.all_equipment):
                if equip.owner == target:
                    self.reserved_equipment[i] = True
        if conditions is None:
            conditions = {}
        for target in conditions:
            if target not in self.target_classes:
                raise Exception(f"Unknown target in conditions '{target}'")
        upgrade_cost = 0
        upgrade_cost_no_accs = 0
        for target in targets.keys():
            if target in protected:
                continue
            target_conditions = []
            if target in conditions:
                target_conditions = conditions[target]
            if add_speed_condition:
                target_conditions.append(["HSPD", AtLeast, 100])
            weights = self.extract_weights(target, targets, target_conditions)
            all_equips, all_stats, all_scores = self.optimize_by_weights(weights, target, upgrade_accs=upgrade_accs)
            is_valid = equipment_is_valid(target_conditions, all_stats[-1])
            if not is_valid:
                all_equips, all_stats, all_scores = self.optimize_with_conditions(
                    target, weights, target_conditions, upgrade_accs)
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

    def extract_weights(self, target, targets, conditions=None):
        """
        Extracts weights for a given target from the given target dict. Throws an exception if targets
        contains invalid information.

        Parameters:
            target (string): Name of the target for which weights are to be read
            targets (dict{string: dict{string, float}}): Maps target names to weights for optimization
            conditions (list[list[string, condition class, int]]): Contains condition stats, typer and thresholds

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
            if stat in targets[target] and stat not in [s for s, _, _ in conditions]:
                weights[i] = targets[target][stat]
        if min(weights) < 0:
            raise Exception(f"Weights for '{target}' contain negative values")
        total = sum(weights)
        return weights if total == 0 else weights / total

    def optimize_by_weights(self, weights, target, cannot_steal=False, protected=None,
                            upgrade_accs=True):
        """
        Finds optimal equipment by maximizing a weighted score of all stats.
        Only considers possible sets of the same material for armor.

        Parameters:
            weights (list[float]): Contains weights for every stat
            target (string): Target character for optimization
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
            if not is_non_builder(weights):
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
            weights (list[float]): Contains weights for every stat
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

    def optimize_with_conditions(self, target, weights, conditions, upgrade_accs=True):
        """
        Finds optimal equipment for each armor material which fulfills all given conditions
        by maximizing a weighted score of all stats.
        Only considers possible sets of the same material for armor.

        Parameters:
            target (string): Target character for optimization
            weights (list[float]): Contains weights for every stat
            conditions (list[list[string, condition class, int]]): Contains condition stats, typer and thresholds
            upgrade_accs (bool): If False: Only use current stats for accessories

        Returns:
            list[list[int]]: Ids of the best equipment for each armor type,
            list[list[int]]: Stats of the best equipment for each armor type,
            list[float]: Scores of the best equipment for each armor type
        """
        # map equip type to list of (equip_id, score) tuples
        slot_dict = {}
        weapon_slots = set()
        for slots in CHAR_SLOTS[self.target_classes[target]]:
            for slot in slots:
                if slot != "Pet":
                    weapon_slots.add(slot)
        for i, equip in enumerate(self.all_equipment):
            if self.reserved_equipment[i] or equip.type != "Armor" and self.armor_only:
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
            score, stats, potential_upgrades = equip.get_weighted_score(weights, upgrade_accs,
                                                                        [stat for stat, func, _ in conditions if
                                                                         func.name == "at_least"])
            if slot not in slot_dict.keys():
                slot_dict[slot] = [(i, score, stats, potential_upgrades)]
            else:
                slot_dict[slot].append((i, score, stats, potential_upgrades))
        # get all slots relevant to this target
        target_slots = ["Hat", "Mask", "Bracers"]
        if self.target_classes[target] == "Squire":
            target_slots.append("Shield")
        if not is_non_builder(weights):
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
                for ((id_0, score_0, _, _), (id_1, score_1, stats_1, cond_fun_1)) in cart_product(obsoletes, obsoletes):
                    if id_0 == id_1:
                        continue
                    if score_0 < score_1:
                        continue
                    is_dominating = True
                    for stat, cond_fun, _ in conditions:
                        if not cond_fun.apply(self.all_equipment[id_0].stat_dict[stat],
                                              self.all_equipment[id_1].stat_dict[stat]):
                            is_dominating = False
                            break
                    if is_dominating:
                        cur_obsoletes.append((id_1, score_1, stats_1, cond_fun_1))
                obsoletes = copy.deepcopy(cur_obsoletes)
            slot_dict[slot] = [(e_id, score, stats, pot_ups) for e_id, score, stats, pot_ups in slot_dict[slot]
                               if e_id not in [i for i, _, _, _ in obsoletes]]
        # get pet and weapon stats for non builders
        own_equip = []
        own_stats = np.zeros_like(weights, dtype=np.int32)
        own_score = 0
        if is_non_builder(weights):
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
            cur_slots = copy.copy(target_slots)
            for pair in cart_product(ARMOR_SLOTS, [material]):
                cur_slots.insert(0, "".join(pair))
            # iterate over cartesian product of equip for slots of target
            combinations = []
            for i in range(self.num_threads):
                combinations.append(islice(
                    cart_product(*[slot_dict[slot] for slot in cur_slots]), i, None, self.num_threads))
            with Pool(self.num_threads) as pool:
                results = pool.map(partial(find_best_combination, conditions=conditions,
                                           own_equip=own_equip, own_score=own_score,
                                           own_stats=own_stats), combinations)
                best_score = -1
                best_equip = []
                best_stats = []
                for cur_equip, cur_score, cur_stats in results:
                    if cur_score > best_score:
                        best_score = cur_score
                        best_equip = cur_equip
                        best_stats = cur_stats

            all_equips.append(best_equip)
            all_scores.append(best_score)
            all_stats.append(best_stats)
        indices = np.argsort(all_scores)
        all_equips = np.array(all_equips)[indices]
        all_stats = np.array(all_stats)[indices]
        all_scores = np.array(all_scores)[indices]
        return all_equips, all_stats, all_scores

    def print_optimization_results(self, all_equips, all_stats, all_scores, weights, target,
                                   print_all=False, upgrade_accs=True):
        """
        Prints equipment found during optimization

        Parameters:
            all_equips (list[list[int]]): Contains indices of equipment found during optimization runs
            all_stats (list[list[float]]): Stats of given equipment
            all_scores (list[float]): Scores of given equipment sets
            weights (list[float]): Contains weights for every stat
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
                    _, item_stats = e.get_weighted_score(weights, upgrade_accs)
                    owner_stats += np.array(item_stats)
                    owner_score += (item_stats * np.array(weights)).sum()
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
