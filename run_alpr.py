import logging
import sys

import cv2
from config.config import Config
from detection.detector import ALPRSystem

import logging


def main():
    Config.init()
    alpr = ALPRSystem()
    try:
        alpr.run()
    except KeyboardInterrupt:
        logging.info("Shutdown requested. Exiting gracefully.")
        cv2.destroyAllWindows()
        sys.exit(0)

if __name__ == "__main__":
    main()