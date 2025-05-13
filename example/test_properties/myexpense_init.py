from kea import *

class Test1(KeaTest):
    @initializer()
    def set_up(self):
        d(resourceId="org.totschnig.myexpenses.debug:id/suw_navbar_next").click()
        d(resourceId="org.totschnig.myexpenses.debug:id/suw_navbar_next").click()
        d(resourceId="org.totschnig.myexpenses.debug:id/suw_navbar_done").click()

    @precondition(lambda self: d(resourceId="itss").exists())
    @rule()
    def search_bar_should_exist_after_rotation(self):
        assert True