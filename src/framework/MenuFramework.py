# Simple menu framework for console menus

# Has a name and holds a value that can be a callable or a Menu object
class MenuOption:
    name: str
    option_value = None # Option value should be a callable or a Menu object

    def __init__(self, option_value, name: str = ""):
        if name == "":
            if option_value is Menu:
                name = f"Go to {option_value.title}"
            elif option_value is callable:
                name = f"Run {str(option_value)}"
        
        self.option_value = option_value
        self.name = name
    
    def get_value(self):
        if callable(self.option_value) or self.option_value is callable:
            return self.option_value

        # If is menu
        return self.option_value.menu_id

# Simple Menu Class
# Shows its options (MenuOption)
class Menu:
    title: str
    menu_id: int

    options: list[MenuOption]
    
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
    
    def show(self):
        menu_string = ""
        menu_string += self.title.center(self.width)

        options_string = ""
        visual_idx_offset = 1

        for idx, option in enumerate(self.options):
            options_string += f"[{idx + visual_idx_offset}]-{option.name}\n"
        
        if self.show_exit_option:
            options_string += f"[0]-{self.exit_option_text}"
        
        menu_string += "\n" + options_string
        print(menu_string)

        return self._wait_for_option_input()

    def _wait_for_option_input(self):
        min_option: int = 1
        if self.show_exit_option:
            min_option = 0
        
        max_option: int = len(self.options)
        
        selected_option = -1
        while selected_option < min_option or selected_option > max_option:
            selected_option = int(input(self.select_option_text))
            # Valid option selected
            if not selected_option < min_option and not selected_option > max_option:
                break
            # Print warning
            print("WARNING: Invalid option!")
        
        if selected_option == 0:
            return False
        
        return self.options[selected_option - 1].get_value()

class MenuHandler:
    main_menu: Menu
    current_menu: Menu

    menu_path: list[Menu]
    menus: dict = {}

    def __init__(self, main_menu: Menu, add_menus: list[Menu]):
        self.main_menu = main_menu
        self.current_menu = main_menu

        self.menu_path = [main_menu]

        for menu in add_menus:
            self.add_menu(menu)

    # Changes current menu and adds the previous to menu_path
    def change_to_menu(self, menu_id: int):
        # If the previous menu is the menu you are trying to go to:
        # Remove it from the menu path (you are going back on the menu path)
        if self.menu_path[-1] == menu_id:
            self.menu_path.pop(-1)
        
        else:
            # Append the previous menu to the menu path
            self.menu_path.append(self.current_menu.menu_id)

        self.current_menu = self.menus[menu_id]

    def add_menu(self, menu: Menu):
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