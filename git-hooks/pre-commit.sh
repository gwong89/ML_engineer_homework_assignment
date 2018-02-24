./git-hooks/pre-commit.py
RESULT=$?
[ $RESULT -ne 0 ] && exit 1
exit 0