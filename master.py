
import os, re, sys

import foreign_names
import surname_detection
import given_name
import places

if __name__ == '__main__':
    ''' hub file that runs all the little files that we worked on
    '''
    foreign_names.main()
    surname = surname_detection.main()
    given_name.main(surname)
    places.main()
