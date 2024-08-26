

import logging
 
logger = logging.getLogger('test_fgsm')
logger.setLevel(logging.DEBUG)
KZT = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s -%(process)s')
KZT.setFormatter(formatter)
logger.addHandler(KZT)

# logging.debug(1)
logger.debug('----调试信息 [debug]------')