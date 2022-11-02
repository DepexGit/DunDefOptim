import os
import zlib
from src.consts import *
from src.equipment import Equipment


def bytes_to_int(byte_list, raw=False):
    """
    Converts list of bytes to int. Uses little endian and an offset of -127.

    Parameters:
        byte_list (bytes): Byte string to convert
        raw (bool): If False: Bytes are signed with offset -127; if True: Bytes are unsigned

    Returns:
        int: Result of conversion
    """
    number = int.from_bytes(byte_list, "little")
    if raw:
        return number
    number -= 127
    if number > 2 ** 31:
        number -= 2 ** 32
    return number


class DataHandler:
    """
    Holds raw save file data and provides methods for parsing it.
    """
    def __init__(self, path=DECOMP_FILE_PATH, compressed=True):
        """
        Constructor for DataHandler class.

        Parameters:
            path (string): Path to save file
            compressed (bool): Indicates whether file at given location is compressed
        """
        try:
            with open(path, "rb") as save_file:
                self.data = save_file.read()
        except FileNotFoundError:
            print (f"Could not open file {path}. Always run 'update' before any other option")
            raise
        self.level_completion_index = 0
        self.last_equipment_index = 0
        if not compressed:
            self.level_completion_index = \
                self.find_next_string_index("CDTARC", 0, limit=len(self.data))[0]
            self.last_equipment_index = self.find_last_equipment_index()

    def decompress_file(self, enable_output=True):
        """
        Decompresses save file at location set in constructor

        Parameters:
            enable_output (bool): If True: write decompressed file to disc

        Returns:
            bytes: Decompressed file
        """
        if not os.path.exists(INPUT_FILE_PATH):
            raise Exception("Could not find save file. Change 'INPUT_FILE_PATH'" +
                            "in src/consts.py to the appropriate location")
        uncompressed_bytes = self.data[0:0]
        zlib_header = [120, 156]
        index, found = self.find_next_string_index(zlib_header, 0, raw=True, limit=len(self.data))
        while found:
            is_valid = True
            try:
                block = zlib.decompress(self.data[index:])
            except zlib.error:
                is_valid = False
            if is_valid:
                uncompressed_bytes += block
            index, found = self.find_next_string_index(zlib_header, index + 1, raw=True, limit=len(self.data))
        if enable_output:
            if not os.path.exists("data"):
                os.mkdir("data")
            with open(DECOMP_FILE_PATH, "wb") as output_file:
                output_file.write(uncompressed_bytes)
        self.data = uncompressed_bytes
        return uncompressed_bytes

    def find_next_string_index(self, string, start_index=0, reverse=False, raw=False, limit=1024):
        """
        Finds next index of given char- or byte string in self.data

        Parameters:
            string (string OR bytes): Search string
            start_index (int): Start index for search region
            reverse (bool): If True, search backwards
            raw (bool): Interpret string as bytes object
            limit (int): Abort search if no result in self.data[start_index: start_index + limit]

        Returns:
            int: Start index of search result, -1 if no result found
            bool: Indicates whether string has been found
        """
        byte_string = [ord(c) for c in string] if not raw else string
        if not reverse:
            search_range = range(start_index, len(self.data))
        else:
            search_range = range(max(0, start_index - len(string)), max(0, start_index - limit), -1)
        for i in search_range:
            found = True
            for j in range(len(byte_string)):
                if self.data[i + j] != byte_string[j]:
                    found = False
                    break
            if found:
                return i, True
        return -1, False

    def extract_string(self, index, length=0):
        """
        Converts the bytes at given index to string, terminates at Null character or at given length

        Parameters
            index (int): Start index of string
            length (int): Max length of string

        Returns:
            string: The extracted string
        """
        ret_string = ""
        i = 0
        running = True
        while running:
            ret_string += chr(self.data[index + i])
            i += 1
            if length == 0:
                if self.data[index + i] == 0:
                    running = False
            else:
                if i >= length:
                    running = False
        return ret_string

    def get_dir_ids(self):
        """
        Creates a dict containing the structure of equipment directories

        Returns:
            dict{int: list[string, int]}: Maps directory id to its name of parent id
        """
        last_index = self.last_equipment_index
        last_index = self.find_next_string_index([0], last_index, raw=True)[0]
        dirs_dict = {int("0xffffffff", 16): ["/", int("0xffffffff", 16)]}
        # offset of 22 might be incorrect, no idea what the bytes before enemy info store
        end_index = self.find_next_string_index("DunDefArchetypes.Enemy", 0)[0] - 22
        index = end_index
        running = True
        while running:
            index = self.find_next_string_index([0] * 2, index, True, True)[0]
            dir_name = self.extract_string(index + 2)
            index -= 4
            dir_id = bytes_to_int(self.data[index - 2: index + 2], True)
            index -= 4
            parent_id = bytes_to_int(self.data[index - 2: index + 2], True)
            dirs_dict[dir_id] = [dir_name, parent_id]
            index -= 4
            if index - 13 < last_index:
                running = False
        return dirs_dict

    def find_last_equipment_index(self):
        """
        Finds the index of the last equipment in self.data

        Returns:
            int: Index of last equipment
        """
        limit_index = self.find_next_string_index(
                "DunDefArchetypes.EnemyGoblin", limit=len(self.data))[0]
        start_index = self.find_next_string_index(
                EQUIPMENT_MARKER, limit_index, reverse=True, raw=True, limit=len(self.data))[0]
        cur_index = start_index + MISC_OFFSET_DICT["name"][0] + 5
        for _ in range(2):
            string_length = bytes_to_int(self.data[cur_index: cur_index + 4], raw=True)
            cur_index += 4 + string_length
        cur_index += 4
        string_length = bytes_to_int(self.data[cur_index: cur_index + 4], raw=True)
        cur_index += 4 + string_length
        return cur_index

    def get_all_equip_stats(self, start_index):
        """
        Initializes equipment with stats stored at given index

        Parameters:
            start_index: Index of equipment in self.data

        Returns:
            Equipment: The created equipment object
        """
        # find id string
        cur_index = start_index + MISC_OFFSET_DICT["name"][0]
        for _ in range(3):
            string_length = bytes_to_int(self.data[cur_index: cur_index + 4], raw=True)
            cur_index += 4 + string_length
        id_start_index = cur_index + 4
        dir_index = id_start_index + bytes_to_int(self.data[cur_index: cur_index + 4], raw=True) + 4
        id_string = self.extract_string(id_start_index)
        equip_type = ""
        type_string = ""
        found = False
        # determine equipment type
        for cur_equip_type in EQUIPMENT_TYPES:
            for cur_type_string in EQUIPMENT_TYPES[cur_equip_type]:
                if cur_type_string in id_string:
                    equip_type = cur_equip_type
                    type_string = cur_type_string
                    found = True
                    break
            if found:
                break
        if not found:
            return None, dir_index + 4
        # init equipment
        if equip_type == "Armor":
            pos_suffix = "ArmorBase_"
            pos_end_index, _ = self.find_next_string_index(pos_suffix, id_start_index)
            pos_start_index = id_start_index + len(type_string)
            armor_pos = self.extract_string(pos_start_index, length=pos_end_index - pos_start_index)
            armor_mat = self.extract_string(pos_end_index + len(pos_suffix))
            ret_equip = Equipment("Armor", armor_pos, armor_mat)
        elif equip_type == "Accessory":
            index = self.find_next_string_index(type_string, id_start_index)[0] + len(type_string)
            tail_string = self.extract_string(index)
            slot = None
            for cur_slot in ACCESSORY_SLOTS:
                for slot_string in ACCESSORY_SLOTS[cur_slot]:
                    if slot_string in tail_string:
                        slot = cur_slot
                        break
            if slot is None:
                raise Exception(f"Unknown accessory slot '{tail_string}' at index {index}")
            ret_equip = Equipment("Accessory", slot)
        elif equip_type == "Familiar":
            ret_equip = Equipment("Familiar", "Pet")
        elif equip_type == "Weapon":
            slot = ""
            for key in WEAPON_NAMES.keys():
                if type_string in WEAPON_NAMES[key]:
                    slot = key
                    break
            ret_equip = Equipment("Weapon", slot)
        else:
            raise Exception(f"Unknown equipment type with id string '{id_string}'")
        ret_equip.index = start_index
        ret_equip.desc_string = id_string
        # init main stats
        equip_stats = {}
        for stat in STAT_OFFSET_DICT.keys():
            offset, length = STAT_OFFSET_DICT[stat]
            equip_stats[stat] = bytes_to_int(
                self.data[start_index + offset: start_index + offset + length])
        ret_equip.stat_dict = equip_stats
        # init misc stats
        offset, length = MISC_OFFSET_DICT["level"]
        ret_equip.level = bytes_to_int(
            self.data[start_index + offset: start_index + offset + length], raw=True)
        offset, length = MISC_OFFSET_DICT["max_upgrades"]
        ret_equip.max_upgrades = bytes_to_int(
            self.data[start_index + offset: start_index + offset + length], raw=True)
        offset, length = MISC_OFFSET_DICT["quality"]
        ret_equip.quality = bytes_to_int(
            self.data[start_index + offset: start_index + offset + length], raw=True)
        offset, length = MISC_OFFSET_DICT["posx"]
        ret_equip.pos_x = bytes_to_int(
            self.data[start_index + offset: start_index + offset + length], raw=True)
        offset, length = MISC_OFFSET_DICT["posy"]
        ret_equip.pos_y = bytes_to_int(
            self.data[start_index + offset: start_index + offset + length], raw=True)
        offset, length = MISC_OFFSET_DICT["posz"]
        ret_equip.pos_z = bytes_to_int(
            self.data[start_index + offset: start_index + offset + length], raw=True)
        # find current position of item
        is_not_used = True if start_index > self.level_completion_index else False
        dir_id = bytes_to_int(self.data[dir_index: dir_index + 4], raw=True)
        if is_not_used and dir_id == int("ffffffff", 16) and \
                ret_equip.pos_x != 0 or ret_equip.pos_y != 0 or ret_equip.pos_z != 0:
            ret_equip.owner = "Tavern Floor"
        elif is_not_used:
            ret_equip.dir_id = dir_id
            ret_equip.owner = ""
        else:
            owner_index = self.find_next_string_index(
                "HeroTemplate", start_index, reverse=True, limit=50000)[0]
            owner_index = self.find_next_string_index([0] * 2, owner_index, True, True)[0]
            owner_index = self.find_next_string_index([0] * 2, owner_index, True, True)[0]
            ret_equip.owner = self.extract_string(owner_index + 2)
        ret_equip.init_upgrades()
        return ret_equip, dir_index + 4

    def get_all_target_classes(self):
        """
        Creates a dict that maps character names to classes

        Returns:
            dict{string: string}: The created dict
        """
        search_string = "HeroTemplate"
        index, found = self.find_next_string_index(search_string, limit=len(self.data))
        classes = {}
        if not found:
            raise Exception("Could not find any characters in save file!")
        delim_index, _ = self.find_next_string_index("DunDefArchetypes.EnemyGoblin", index, limit=len(self.data))
        stop = False
        while index < delim_index and found:
            class_id = self.extract_string(index + len(search_string))
            class_name = ""
            if class_id == "Recruit" or class_id == "Monkette":
                class_name = "Monk"
            elif class_id == "Apprentice" or class_id == "Sorceress":
                class_name = "Mage"
            elif class_id == "Squire" or class_id == "LadyKnight":
                class_name = "Squire"
            elif class_id == "Initiate" or class_id == "Hunter":
                class_name = "Hunter"
            elif class_id == "RobotGirl":
                class_name = "SeriesEV"
            elif class_id == "Jester":
                class_name = "Jester"
            elif class_id == "Summoner":
                class_name = "Summoner"
            elif class_id == "Barbarian":
                class_name = "Barbarian"
            else:
                raise Exception(f"Unknown character class '{class_id}'")
            name_index, _ = self.find_next_string_index([0] * 2, index, raw=True, reverse=True)
            name_index, _ = self.find_next_string_index([0] * 3, name_index, raw=True, reverse=True)
            name = self.extract_string(name_index + 3)
            if name in classes.keys():
                raise Exception(f"Multiple characters with name '{name}'")
            classes[name] = class_name
            index, found = self.find_next_string_index(search_string, index + 1, limit=len(self.data))
        return classes
