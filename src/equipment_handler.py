import numpy as np
import multiprocessing
import os
import math
from prettytable import PrettyTable
from colorama import Fore, Back, Style
from src.consts import *
from src.data_handler import DataHandler
from src.equipment import Equipment


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
                if cur_equip.owner == "Tavern Floor":
                    on_floor = True
                if on_floor and cur_equip.dir_id == 2 ** 32 - 1 and \
                        cur_equip.pos_x == 0 and cur_equip.pos_y == 0 and cur_equip.pos_z == 0:
                    cur_equip.owner = "Shop"
                    cur_equip.dir_id = -1
                found_equip.append(cur_equip)
        return found_equip

    def optimize_for_targets(self, targets, print_all=False):
        """
        Finds a set of equipment for each given target by optimizing for given weights.
        Targets have descending priority for equipment by their order in given targets dict.
        Assigns one item per slot of target character if target is a builder. Ignores weapons and
        pets for non builders.

        Parameters:
            targets (dict{string: dict{string, float}}): Maps target names to weights for optimization
        """
        for target in targets.keys():
            if target not in [e.owner for e in self.all_equipment]:
                raise Exception(f"Unknown optimization target '{target}'")
            for key in targets[target].keys():
                if key not in STAT_OFFSET_DICT.keys():
                    raise Exception(f"Unknown stat '{key}'")
            weights = [0] * (len(STAT_OFFSET_DICT.keys()) - 4)
            for i, stat in enumerate(list(STAT_OFFSET_DICT.keys())[4:]):
                if stat in targets[target]:
                    weights[i] = targets[target][stat]
            equips, score = self.optimize_by_weights(weights, target, print_all)
            for e in equips:
                self.reserved_equipment[e] = True

    def optimize_by_weights(self, weights, target, print_all=False, cannot_steal=False, protected=[]):
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

        Returns:
            list[int]: Indices of equipment found during optimization
            float: Optimization score
        """
        print("Finding optimal equipment...")
        all_equips = []
        all_stats = []
        all_scores = []
        # armors
        for equip_mat in ARMOR_MATERIALS:
            all_equips, all_stats, all_scores = self.optimize_for_slots(
                ARMOR_SLOTS, weights, target, cannot_steal, protected,
                all_equips, all_stats, all_scores, equip_mat)
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
                all_equips, all_stats, all_scores)
            if max(weights[:4]) <= 0:
                # pets and weapons
                all_equips, all_stats, all_scores = self.optimize_for_slots(
                    CHAR_SLOTS[self.target_classes[target]], weights, target, cannot_steal, protected,
                    all_equips, all_stats, all_scores)
        if target in self.print_targets:
            self.print_optimization_results(all_equips, all_stats, all_scores, weights, target, print_all)
        return all_equips[-1], all_scores[-1]

    def optimize_for_slots(self, slots, weights, target, cannot_steal, protected,
                           prev_equips, prev_stats, prev_scores, equip_mat=None):
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
                cur_score, cur_stats = equipment.get_weighted_score(weights)
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

    def print_optimization_results(self, all_equips, all_stats, all_scores, weights, target, print_all=False):
        """
        Prints equipment found during optimization

        Parameters:
            all_equips (list[list[int]]): Contains indices equipment found during optimization runs
            all_stats (list[list[float]]): Stats of given equipment
            prev_scores (list[float]): Scores of given equipment sets
            weights (dict{string: float}): Maps stat name to weight
            target (string): Target character for optimization
            print_all (bool): Print best equipment set for each armor material instead only the top one
        """
        def pos_neg_color(num):
            color = Fore.YELLOW
            if num < 0:
                color = Fore.RED
            elif num > 0:
                color = Fore.GREEN
            return color
        delim = "=========================================================="
        header = list(STAT_OFFSET_DICT.keys())[4:]
        start_index = 0
        is_dps_target = False
        if max(weights[:4]) > 0:
            is_dps_target = True
        if not print_all:
            start_index = len(all_scores) - 1
        for i in range(start_index, len(all_scores)):
            equip, stat, score = all_equips[i], all_stats[i], all_scores[i]
            score = math.ceil(score)
            print(delim)
            print(f"Rank {len(all_scores) - i} ({self.all_equipment[equip[0]].material})" + ":\n" +
                  self.generate_table(equip).__str__())
            p_table = PrettyTable(header)
            stat = list(stat)
            if not self.raw_output:
                for i, s in enumerate(stat):
                    if weights[i] > 0:
                        stat[i] = Style.BRIGHT + str(s) + Style.RESET_ALL
                    else:
                        stat[i] = Style.DIM + str(s) + Style.RESET_ALL
            p_table.add_row(list(stat))
            print(f"Stats with score {score}:\n" + p_table.__str__())
            if target != "":
                owner_stats = np.zeros(len(weights))
                owner_score = 0
                for e in self.all_equipment:
                    if e.owner == target and not \
                            (is_dps_target and (e.type == "Weapon" or e.type == "Familiar")) and not \
                            (self.armor_only and e.type != "Armor"):
                        item_score, item_stats = e.get_weighted_score(weights)
                        owner_stats += np.array(item_stats)
                        owner_score += item_score
                owner_score = math.ceil(owner_score)
                stat_diffs = [int(e) for e in list(np.array(all_stats[-1]) - owner_stats)]
                if not self.raw_output:
                    for i in range(len(stat_diffs)):
                        if weights[i] > 0:
                            stat_diffs[i] = pos_neg_color(stat_diffs[i]) + str(stat_diffs[i]) + Style.RESET_ALL
                    p_table = PrettyTable(header)
                    p_table.add_row(stat_diffs)
                    target_str = target
                    if not self.raw_output:
                        target_str = Fore.BLUE + Style.BRIGHT + target + Style.RESET_ALL
                    print(f"Stat changes for {target_str} with score delta " +
                          f"{pos_neg_color(score - owner_score) + str(score - owner_score) + Style.RESET_ALL}:\n" +
                          p_table.__str__())
        print(delim)

    def generate_table(self, indices, sort=False):
        """
        Generates PrettyTable for equipment with given indices

        Parameters:
            indices (list[int]): Indices for equipment

        Returns:
            PrettyTable: Contains stats for given equipment
        """
        header = ["Type", "Slot", "Material", "Tier"] + list(STAT_OFFSET_DICT.keys())[4:] + \
                 ["Level", "Location"]
        p_table = PrettyTable(header)
        if sort:
            all_equip = sorted(self.all_equipment, key=lambda e: e.index)
        else:
            all_equip = self.all_equipment
        for i in indices:
            p_table.add_row(all_equip[i].get_property_list(self.dirs))
        if self.raw_output:
            return p_table.__str__()
        table = ""
        for i, row in enumerate(p_table.__str__().split("\n")):
            if i > 2 and i % 2 == 0 and row[0] != "+":
                row = row[0] + Back.LIGHTBLACK_EX + row[1:-1] + Style.RESET_ALL + row[-1]
            table += row + "\n"
        table = table[:-1]
        return table

    def __str__(self):
        """
        Generates PrettyTable containing every equipment in self.all_equipment

        Returns:
            string: String form of generated PrettyTable
        """
        table_string = self.generate_table(list(range(len(self.all_equipment))), sort=True)
        type_counts = {}
        for equip_type in EQUIPMENT_TYPES.keys():
            type_counts[equip_type] = 0
        for equip in self.all_equipment:
            for equip_type in EQUIPMENT_TYPES.keys():
                if equip.type == equip_type:
                    type_counts[equip.type] += 1
                    break
        p_table = PrettyTable(type_counts.keys())
        p_table.add_row(type_counts.values())
        return table_string + "\nTotals:\n" + p_table.__str__()
