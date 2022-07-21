''' Dumps a config file of the type readable by configparser
into a dictionary 
Ref: http://docs.python.org/library/configparser.html

Downloaded from git@gist.github.com:c5607843dd7174b8d6828185a09584e7.git
'''

import sys
import configparser

class GetDict:

    def __init__(self, config):
        self.config = config

    def get_dict(self):
        # config = configparser.ConfigParser()
        # config.read(self.config)

        sections_dict = {}

        # get all defaults
        defaults = self.config.defaults()
        temp_dict = {}
        for key in defaults.keys():
            temp_dict[key] = defaults[key]

        sections_dict['default'] = temp_dict

        # get sections and iterate over each
        sections = self.config.sections()

        for section in sections:
            options = self.config.options(section)
            temp_dict = {}
            for option in options:
                temp_dict[option] = self.config.get(section,option)

            sections_dict[section] = temp_dict

        return sections_dict

if __name__== '__main__':

    if len(sys.argv) == 1:
        print('Must provide the path to the config file as the argument')
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    getdict = GetDict(config)
    config_dict = getdict.get_dict()

    # print the entire dictionary
    # Trick from http://stackoverflow.com/a/3314411/59634
    import json
    print(json.dumps(config_dict, indent=2))
