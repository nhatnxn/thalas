import connexion
from swagger_server.models.classnames import Classnames  # noqa: E501
from swagger_server.inference.serve import get_model_thalas, get_model_parameters
from swagger_server.models.body_detect import BodyDetect
from swagger_server.models.value import Value  # noqa: E501
from swagger_server.models.key_value import KeyValue


def make_parameters_report(cnt):
    key_value = list()
    for id in range(cnt.shape[0]):
        if int(id) == 0:
            continue
        if cnt.iloc[id][1] is None or len(cnt.iloc[id][1]) == 0:
            cnt.iloc[id][1] = -1
        try:
            key_value.append(KeyValue(key=cnt.iloc[id][0], value=float(cnt.iloc[id][1])))
        except:
            key_value.append(KeyValue(key=cnt.iloc[id][0], value=-1.0))
    return Value(parameters=key_value)


def check_input(corners):
    for corner in corners:
        if corner.x < 0 or corner.y < 0:
            return False
    return True


def detect(body=None):  # noqa: E501  # noqa: E501
    """detect

    Detect values in table # noqa: E501

    :param file:
    :type file: strstr
    :param image_url:
    :type image_url: str
    :param borders:
    :type borders: str

    :rtype: Value
    """
    if connexion.request.is_json:
        body = BodyDetect.from_dict(connexion.request.get_json())
    p = get_model_parameters()
    check = check_input(body.corners)
    if check:
        result = p.transform({"imageUrl": body.image_url, 'corners': [(corner.y, corner.x) for corner in body.corners]})
    else:
        result = p.transform({"imageUrl": body.image_url})
    if isinstance(result, str):
        return Value(error_code=result)
    return make_parameters_report(result)


def detect_by_thalas(body=None):  # noqa: E501
    """detect_by_thalas

    Detect the thalas # noqa: E501

    :param body: 
    :type body: List[]

    :rtype: StringArray
    """
    p = get_model_thalas()
    if body.get("MODE") == "9PARAMS":
        result = p.predict_9param(body)
    elif body.get("MODE") == "11PARAMS":
        result = p.predict_11param(body)
    else:
        result = p.predict_13param(body)

    return Classnames(classnames=result)


def health_check():  # noqa: E501
    """health_check

     # noqa: E501


    :rtype: str
    """
    return 'OK'
