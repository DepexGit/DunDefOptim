import os
import re
import argparse
import pickle
from colorama import init as init_colors
from src.consts import *
from src.settings import INPUT_FILE_PATH, OPTIM_TARGETS
from src.equipment_handler import EquipmentHandler
from src.data_handler import DataHandler
from src.hotkeys import setup_hotkeys

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    # cli argument parser setup
    stat_string = ""
    for key in list(STAT_OFFSET_DICT.keys())[4:]:
        stat_string += key + ", "
    stat_string = stat_string[:-2]
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument("mode", choices=["update", "print", "find_obsoletes", "hotkeys", "optimize", "debug"],
                            help="Selects task to execute:\n" +
                                 "update:           Decompress save file from game folder." +
                                 "Use 'Export to open' in-game before running this command\n" +
                                 "print:            Print a list of all equipment in save file.\n" +
                                 "find_obsoletes:   List all pareto dominated equipment.\n" +
                                 "hotkeys:          Start tower stacking script.\n" +
                                 "optimize:         Find optimal equipment from save file." +
                                 "Modify 'OPTIM_TARGETS' in src/settings.py or\n" +
                                 "                  use -w and -t to set optimization parameters")
    arg_parser.add_argument("-r", "--raw", action="store_true", help="Only output ASCII characters")
    arg_parser.add_argument("-t", "--target", help="Select character as optimization target")
    arg_parser.add_argument("-w", "--weights", nargs="+",
                            help=f"Specify weights for stats. Stat names are: {stat_string}")
    arg_parser.add_argument("-f", "--full", action="store_true", help="Print best combination for each material")
    arg_parser.add_argument("-n", "--num_threads", type=int, default=-1, help="Number of worker threads to use " +
                                                                              "while running update")
    arg_parser.add_argument("-a", "--armor_only", action="store_true", help="Only optimize armors")
    arg_parser.add_argument("-N", "--no_accessory_upgrades", action="store_true",
                            help="Do not upgrade accessories, use current stats instead")
    arg_parser.add_argument("-p", "--print_targets", nargs="+", help="Only output optimization results" +
                                                                     " for given targets")
    arg_parser.add_argument("-P", "--print_types", nargs="+", help="Only output equipment matching the given types")
    arg_parser.add_argument("-e", "--export_csv", action="store_true", help="Print table as csv, implies --raw")
    args = arg_parser.parse_args()

    init_colors(autoreset=True)
    if args.mode == "update":
        print("Decompressing file...")
        dh = DataHandler(INPUT_FILE_PATH)
        dh.decompress_file()
        print("Initializing equipment...")
        eh = EquipmentHandler(num_threads=args.num_threads)
        with open(EQUIP_FILE_PATH, "wb") as equip_file:
            pickle.dump(eh, equip_file)
        exit(0)

    try:
        with open(EQUIP_FILE_PATH, "rb") as equip_file:
            eh = pickle.load(equip_file)
    except FileNotFoundError:
        print(f"Could not open {EQUIP_FILE_PATH}. Always run 'update' before any other option\n")
        raise
    eh.raw_output = args.raw
    eh.armor_only = args.armor_only
    eh.print_csv = args.export_csv
    if args.print_types is not None:
        type_filter = []
        for equip_type in args.print_types:
            if equip_type not in EQUIPMENT_TYPES.keys():
                arg_parser.error(f"Unknown equipment type '{equip_type}'")
            type_filter.append(equip_type)
        eh.print_type_filter = type_filter
    if args.mode == "print":
        print(eh)
    elif args.mode == "find_obsoletes":
        eh.find_obsolete_equipment()
    elif args.mode == "hotkeys":
        setup_hotkeys()
    elif args.mode == "optimize":
        if args.target and args.weights:
            weight_dict = {}
            cur_stat = ""
            for i, elem in enumerate(args.weights):
                if i % 2 == 0:
                    if elem not in list(STAT_OFFSET_DICT.keys())[4:]:
                        arg_parser.error(f"Unknown stat '{elem}'")
                    cur_stat = elem
                else:
                    if not re.match(r"\d+(\.\d*)?", elem):
                        arg_parser.error(f"Weights must be numbers. '{elem}' is not a number")
                    weight_dict[cur_stat] = float(elem)
            eh.optimize_for_targets({args.target: weight_dict}, args.full, not args.no_accessory_upgrades)
        elif not args.target and not args.weights:
            if args.print_targets is not None:
                for target in args.print_targets:
                    if target not in list(OPTIM_TARGETS.keys()):
                        raise Exception(f"Unknown target '{target}'")
                eh.print_targets = args.print_targets
            eh.optimize_for_targets(OPTIM_TARGETS, args.full, not args.no_accessory_upgrades)
        else:
            arg_parser.error("Optimizing with custom options requires weights and a target character")
    elif args.mode == "debug":
        print(eh.target_classes)
