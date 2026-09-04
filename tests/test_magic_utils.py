from eregion.utils.dangerous_magic import pack_argument_helper


def test_positional_args_call():

    def f(parg1, parg2):
        return pack_argument_helper()

    a = f(1, 2)

    assert "parg1" in a
    assert "parg2" in a

    assert a["parg1"] == 1
    assert a["parg2"] == 2


def test_mixed_args_call():

    def f(parg1, parg2, kwarg1, kwarg2):
        return pack_argument_helper()

    a = f(1, 2, "one", "two")

    assert a == {"parg1": 1, "parg2": 2, "kwarg1": "one", "kwarg2": "two"}


def test_forced_kwargs_call():
    def f(parg1, *, kwarg1):
        return pack_argument_helper()

    a = f(1, kwarg1="one")

    assert a == {"parg1": 1, "kwarg1": "one"}


def test_instance_method_args_call():
    class A:
        def f(self, parg1, kwarg1):
            return pack_argument_helper(selfarg=self)

    a = A()
    kw = a.f(1, kwarg1="one")

    assert kw == {"parg1": 1, "kwarg1": "one"}


def test_class_method_args_call():
    class A:
        @classmethod
        def g(cls, parg1, kwarg1):
            return pack_argument_helper(selfarg=cls)

    kw = A.g(1, kwarg1="one")
    assert kw == {"parg1": 1, "kwarg1": "one"}


def test_excludeargs_call():
    def f(parg1, parg2):
        return pack_argument_helper(exclude_args=["parg1"])

    kw = f(1, 2)
    assert "parg1" not in kw


def test_kwargs_args_call():

    def f(parg1, parg2, kwarg1, **kwargs):
        return pack_argument_helper()

    kw = f(1, 2, "one", kwarg2="two")
    assert kw == {"parg1": 1, "parg2": 2, "kwarg1": "one", "kwarg2": "two"}
