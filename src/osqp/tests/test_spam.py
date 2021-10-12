from osqp.spam import system


def test_whoami():
    status = system('whoami')
    assert status == 0


def test_whoareyou():
    status = system('whoareyou')
    assert status != 0