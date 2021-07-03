# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server import util


class ImageUrl(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    def __init__(self, image_url: str=None):  # noqa: E501
        """ImageUrl - a model defined in Swagger

        :param image_url: The image_url of this ImageUrl.  # noqa: E501
        :type image_url: str
        """
        self.swagger_types = {
            'image_url': str
        }

        self.attribute_map = {
            'image_url': 'imageUrl'
        }
        self._image_url = image_url

    @classmethod
    def from_dict(cls, dikt) -> 'ImageUrl':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The imageUrl of this ImageUrl.  # noqa: E501
        :rtype: ImageUrl
        """
        return util.deserialize_model(dikt, cls)

    @property
    def image_url(self) -> str:
        """Gets the image_url of this ImageUrl.


        :return: The image_url of this ImageUrl.
        :rtype: str
        """
        return self._image_url

    @image_url.setter
    def image_url(self, image_url: str):
        """Sets the image_url of this ImageUrl.


        :param image_url: The image_url of this ImageUrl.
        :type image_url: str
        """

        self._image_url = image_url
