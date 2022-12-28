# encoding: utf-8

from .center_loss import CenterLoss
from .gem_pool import GeneralizedMeanPooling, GeneralizedMeanPoolingP
from .non_local import Non_local
from .triplet_loss import (CrossEntropyLabelSmooth, TripletLoss,
                           WeightedRegularizedTriplet)
