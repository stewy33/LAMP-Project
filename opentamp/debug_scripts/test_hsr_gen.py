import openravepy
from opentamp.core.util_classes.openrave_body import OpenRAVEBody
from opentamp.core.util_classes.viewer import OpenRAVEViewer
from opentamp.core.util_classes.robots import *

env = openravepy.Environment()
body = OpenRAVEBody(env, 'hsr', HSR())
env.SetViewer('qtcoin')
