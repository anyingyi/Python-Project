#     Copyright (c) 2020. Huawei Technologies Co., Ltd.
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

import os
import sys


class Pack:
    def __init__(self, input_str: str):
        """
        Initialization of packing algorithm.
        :param input_str: the content of the input file.
        """
        self.input_str = input_str

    def run(self) -> str:
        """
        Generate the output JSON string.
        :return Output JSON string.
        """
        raise NotImplementedError


def save(output_str, output_dir, file_name):
    """
    Save the result to a file named after the input file.
    :param output_str: the output JSON string.
    :param output_dir: the output directory.
    :param file_name: the output file name, same as the input file.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, file_name)
    with open(
            output_path, 'w', encoding='utf-8', errors='ignore') as f:
        json.dump(output_str, f, ensure_ascii=False)


def main(argv):
    input_dir = argv[1]
    output_dir = argv[2]
    for file_name in os.listdir(input_dir):
        # Read the input file.
        input_path = os.path.join(input_dir, file_name)
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            message_str = f.read()
        # Run the packing algorithm and get the output JSON string
        pack = Pack(message_str)
        output_str = pack.run()
        # Save the result to a file named after the input file.
        save(output_str, output_dir, file_name)


if __name__ == "__main__":
    main(sys.argv)
