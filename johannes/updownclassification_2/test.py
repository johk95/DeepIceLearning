#!/usr/bin/env python
# coding: utf-8

# this script uses theos datasets. That is a try to reduce overfitting

import os
import sys
import numpy as np
import argparse

## constants ##
energy, azmiuth, zenith, muex = 0, 1, 2, 3

################# Function Definitions ####################################################################

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="The name for the Project", type=str ,default='updown_NN')
    parser.add_argument("--input", help="Name of the input files seperated by :", type=str ,default='all')
    parser.add_argument("--model", help="Name of the File containing the model", type=str, default='FCNN_v1.cfg')
    parser.add_argument("--using", help="charge or time", type=str, default='time')
    parser.add_argument("--virtual_len", help="Use an artifical array length (for debugging only!)", type=int , default=-1)
    parser.add_argument("--continue", help="Give a folder to continue the training of the network", type=str, default = 'None')
    parser.add_argument("--date", help="Give current date to identify safe folder", type=str, default = 'None')
    parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')
    parser.add_argument("--filesizes", help="Print the number of events in each file and don't do anything else.", nargs='?',
                       const=True, default=False)
    parser.add_argument("--testing", help="loads latest model and just does some testing", nargs='?', const=True, default=False)
    parser.add_argument("--crtfolders", help="creates the folderstructure so you can redirect nohup output to it. take care of day-change: at 7'o'clock in the morning (german summer time).", nargs='?', const=True, default=False)
      # Parse arguments
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parseArguments()
    print args.__dict__