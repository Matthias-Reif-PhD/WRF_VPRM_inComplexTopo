import json
import jwt
import time

# Load saved key from filesystem
service_key = json.load(open("/home/c707/c7071034/Github/WRF_VPRM_post/pmodel/get_modis_data/clms_key.json", "rb"))

private_key = service_key["private_key"].encode("utf-8")

claim_set = {
    "iss": service_key["client_id"],
    "sub": service_key["user_id"],
    "aud": service_key["token_uri"],
    "iat": int(time.time()),
    "exp": int(time.time() + (60 * 60)),
}
grant = jwt.encode(claim_set, private_key, algorithm="RS256")
