import sys
import json

import gcnu_common.utils.config_dict


def main(argv):

    estInitNumber = 99999999

    estInitConfigFilenamePattern = \
        "../data/{:08d}_estimation_metaData.{{:s}}".format(estInitNumber)
    estInitConfigFilename = estInitConfigFilenamePattern.format("ini")
    estInitConfigTxtFilename = estInitConfigFilenamePattern.format("txt")
    config_file_params = gcnu_common.utils.config_dict.GetDict(config=estInitConfigFilename).get_dict()

    print(config_file_params)
    config_file_params_str = json.dumps(config_file_params, indent=2)
    with open(estInitConfigTxtFilename, "w") as f:
        f.writelines(config_file_params_str)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
