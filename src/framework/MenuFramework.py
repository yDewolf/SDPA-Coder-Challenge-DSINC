# Simple menu framework for console menus

# Has a name and holds a value that can be a callable or a Menu object
class MenuOption:
    name: str
    option_value = None # Option value should be a callable or a Menu object
    has_visibility_condition: bool = False

    # Runs this callable to check if this option should be visible or not
    # Receives Menu Handler as a parameter
    visibiltity_condition: callable = None
    inverse_condition: bool = False


    def __init__(self, option_value, name: str = "", visibility_condition: callable = None, inverse_condition: bool = False):
        if name == "":
            if type(option_value) is Menu:
                name = f"Go to {option_value.title}"
            elif type(option_value) is callable:
                name = f"Run {str(option_value)}"
            elif type(option_value) is str:
                name = f"Select {option_value}"
        
        if visibility_condition != None:
            self.add_visibiltiy_condition(visibility_condition)
            self.inverse_condition = inverse_condition

        self.option_value = option_value
        self.name = name
    
    def get_value(self):
        if callable(self.option_value) or self.option_value is callable:
            return self.option_value

        if type(self.option_value) is str:
            return self.option_value

        # If is menu
        return self.option_value.menu_id

    def add_visibiltiy_condition(self, visibility_condition_callable: callable):
        self.has_visibility_condition = True
        self.visibiltity_condition = visibility_condition_callable

# Simple Menu Class
# Shows its options (MenuOption)
class Menu:
    title: str
    menu_id: int
    menu_handler: object

    options: list[MenuOption]
    
    # Should return a list[options]
    # Receives self.options
    # Also receives menu handler
    update_options: callable

    # Should return a str
    # Receives the current title
    # Also receives menu handler
    update_title: callable
    
    # Receives the selected option value
    # Also receives menu handler
    call_custom_function: callable 

    show_exit_option: bool = True
    exit_option_text: str = "Go back"
    
    select_option_text: str = "Select one of the options above: "
    width: int = 75

    def __init__(self, menu_id: int, menu_title: str, menu_options: list[MenuOption], width: int = 75, show_exit_option: bool = True):
        self.menu_id = menu_id
        self.title = menu_title
        self.options = menu_options

        self.width = width
        self.show_exit_option = show_exit_option

        self.update_options = None
        self.update_title = None
        self.call_custom_function = None
    
    def show(self):
        if self.update_title:
            self.title = self.update_title(self.title, self.menu_handler)
        
        if self.update_options:
            self.options = self.update_options(self.options, self.menu_handler)

        menu_string = ""
        menu_string += apply_color(self.title, "header").center(self.width)

        options_string = ""
        visual_idx_offset = 1

        available_options: list[MenuOption] = []

        for option in self.options:
            if option.has_visibility_condition:
                condition_match = option.visibiltity_condition(self.menu_handler)
                if option.inverse_condition:
                    condition_match = not condition_match

                if not condition_match:
                    continue
            
            available_options.append(option)


        for idx, option in enumerate(available_options):
            options_string += f"[{(idx + visual_idx_offset)}]-{option.name}\n"
        
        if self.show_exit_option:
            options_string += f"[{apply_color("0", "red")}]-{self.exit_option_text}"
        
        menu_string += "\n" + options_string
        print(menu_string)

        option_value = self._wait_for_option_input(available_options)

        # Call custom function only if option value is not False
        if self.call_custom_function and option_value:
            # Call custom function and then go back to previous menu
            self.call_custom_function(option_value, self.menu_handler)
            return False

        return option_value


    def _wait_for_option_input(self, available_options: list[MenuOption]):
        min_option: int = 1
        if self.show_exit_option:
            min_option = 0
        
        max_option: int = len(available_options)
        
        selected_option = range_input_value(min_option, max_option, self.select_option_text)
        
        if selected_option == 0:
            return False
        
        return available_options[selected_option - 1].get_value()

class MenuHandler:
    main_menu: Menu
    current_menu: Menu

    menu_path: list[Menu]
    menus: dict = {}
    global_variables = {}

    def __init__(self, main_menu: Menu, search_for_menus: bool = True, add_menus: list[Menu] = []):
        self.main_menu = main_menu
        self.current_menu = main_menu

        self.menu_path = [main_menu]

        if search_for_menus:
            add_menus = _get_deep_menus(main_menu)
        
        for menu in add_menus:
            self.add_menu(menu)

    # Changes current menu and adds the previous to menu_path
    def change_to_menu(self, menu_id: int):
        if not self.menus.__contains__(menu_id):
            # Push Error:
            print(f"ERROR: Menu id wasn't found in menus | Provided ID: {menu_id} | Menus: {self.menus}")
            return

        # If the previous menu is the menu you are trying to go to:
        # Remove it from the menu path (you are going back on the menu path)
        if self.menu_path[-1] == menu_id:
            self.menu_path.pop(-1)
        
        else:
            # Append the previous menu to the menu path
            #self.current_menu.global_variables = self.global_variables
            self.menu_path.append(self.current_menu.menu_id)

        # Update global variables for the next menu
        #self.menus[menu_id].global_variables = self.global_variables

        self.current_menu = self.menus[menu_id]

    def add_menu(self, menu: Menu):
        #menu.global_variables = self.global_variables
        menu.menu_handler = self
        self.menus[menu.menu_id] = menu

    ## Main loop of this MenuHandler
    def main_loop(self):
        while True:
            selected_option_value = self.current_menu.show()
            # Quit or Go back
            if selected_option_value == False:
                # If the selected option isn't any of the previous checks
                # if the current menu is already the main menu: leave from loop
                if self.current_menu == self.main_menu:
                    break
                
                # if the current menu is not the main menu: go back to previous menu
                else:
                    self.change_to_menu(self.menu_path[-1])
                    continue
            
            # Run callable
            if callable(selected_option_value):
                selected_option_value()
                if self.current_menu == self.main_menu:
                    continue
                # Go to previous menu
                self.change_to_menu(self.menu_path[-1])
                continue
            
            # Go to the selected menu
            self.change_to_menu(selected_option_value)
            continue

# Looks at menu options
# When it founds a MenuOption that references a Menu, it appends this Menu to menus list
# This function is executed recursively
def _get_deep_menus(menu: Menu) -> list[Menu]:
    menus: list[Menu] = [menu]
    for option in menu.options:
        if type(option.option_value) != Menu:
            continue

        # Append the menu that was found
        menus.append(option.option_value)
        # Look for menus inside the menu that was found
        menus += _get_deep_menus(option.option_value)

    return menus


def range_input_value(min_value: int, max_value: int, input_text: str):
    value: int = -1
    while value <= min_value or value >= max_value:
        value = input(input_text)
        if not value.isnumeric(): value = min_value - 1
        value = int(value)

        if not value <= min_value or not value >= max_value:
            break
            
        # Print Warning
        print("WARNING: Invalid option!")
    
    return int(value)

def bool_input_value(input_text: str):
    #value = -1
    #print("[1]-Yes\n[0]-No")
    value = range_input_value(0, 1, input_text + f"\n{get_str_as_color("green")}[1]-Yes\n{get_str_as_color("red")}[0]-No{get_str_as_color("endc")}\n")

    return bool(value)

def list_input_value(valid_values: list[int], input_text: str, invalid_value: int = -1):
    value = invalid_value

    while not valid_values.__contains__(value):
        value = input(input_text)
        if not value.replace("-","").isnumeric(): value = invalid_value
        value = int(value)

        if valid_values.__contains__(value):
            break
            
        # Print Warning
        print("WARNING: Invalid option!")
    
    return value



def get_str_as_color(color_str: str):
    colors = {
        "red": '\033[91m',
        "green": '\033[92m',
        "orange": '\033[93m',
        "blue": '\033[94m',
        "header": "\033[95m",
        "cyan": '\033[96m',
        "endc": '\033[0m'
    }

    # Propositalmente vai dar erro se você colocar uma cor que não tá na lista de cores
    # Tem formas melhoras de fazer isso, mas faz parte
    return colors[color_str]

def apply_color(text: str, color: str):
    return get_str_as_color(color) + text + get_str_as_color("endc")

# Color options: red, blue, orange, green
def print_colored(text: str, color: str):
    print(apply_color(text, color))

def push_error(text: str):
    print(apply_color("ERROR: " + text, "red"))

def push_warning(text: str):
    print(apply_color("WARNING: " + text, "orange"))



# Example usage:
#test_menu3 = Menu(3, "Test menu 3", [MenuOption(lambda: print("This is a test function from test menu 3"), "Run a lambda function")])
#
#test_menu2 = Menu(2, "Test menu 2 (from Test menu 1)", [MenuOption(test_menu3)])
#test_menu1 = Menu(1, "Test menu 1", [MenuOption(test_menu2, "Go to Test menu 2")], show_exit_option=True)
#
#
#main_menu = Menu(0, "Main menu", [MenuOption(test_menu1, "Go to test menu 1"), MenuOption(test_menu3, "Go to test menu 3")])
#main_menu.exit_option_text = "Leave"
#
#menu_handler = MenuHandler(main_menu, [test_menu3, test_menu2, test_menu1, main_menu])
#
#menu_handler.main_loop()