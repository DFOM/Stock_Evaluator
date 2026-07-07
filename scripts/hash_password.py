#!/usr/bin/env python3
"""Generate the bcrypt hash for ADMIN_PASSWORD_HASH.

    python scripts/hash_password.py            # prompts (input hidden)
    python scripts/hash_password.py mypassword # direct (visible in shell history!)
"""
import getpass
import sys

import bcrypt


def main() -> None:
    if len(sys.argv) > 1:
        password = sys.argv[1]
    else:
        password = getpass.getpass("Password to hash: ")
        if password != getpass.getpass("Repeat: "):
            print("Passwords do not match.", file=sys.stderr)
            raise SystemExit(1)
    if len(password) < 8:
        print("Use at least 8 characters.", file=sys.stderr)
        raise SystemExit(1)
    print(bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode())


if __name__ == "__main__":
    main()
