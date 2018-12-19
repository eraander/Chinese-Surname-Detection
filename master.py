
import os, re, sys

import foreign_names
import surname_detection
import given_name
import places

if __name__ == '__main__':
    ''' hub file that runs all the little files that we worked on
    ''' 

    print('\nPERSONAL NAMES: \n')
    surname = surname_detection.main()
    given_name.main(surname)
    print('\nPLACE NAMES: \n')
    places.main()
    print('\nFOREIGN NAMES: \n')
    foreign_names.main()
