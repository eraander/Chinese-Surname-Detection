
import os, re, sys

import foreign_names
import surname_detection
import places

if __name__ == '__main__':
    ''' hub file that runs all the little files that we worked on
    '''
    foreign_names.main()
    surname_detection.main()
    places.main()