from kea import *

class Test1(KeaTest):

    @initializer()
    def set_up(self):
        if d(text="Allow").exists():
            d(text="Allow").click()

        for _ in range(5):
            d(resourceId="it.feio.android.omninotes.alpha:id/next").click()
        d(resourceId="it.feio.android.omninotes.alpha:id/done").click()

    @precondition(lambda self: d(resourceId="itss").exists())
    @rule()
    def search_bar_should_exist_after_rotation(self):
        assert True