import sys
sys.path.append("..")
from kea import *

class Test(KeaTest):
    @mainPath()
    def delete_activity_mainpath(self):
        d(description="Open navigation").click()
        d(text="Edit Activities").click()

    @precondition(
        lambda self: d(text="Manage activities").exists() and d(resourceId="de.rampro.activitydiary:id/action_show_hide_deleted").exists() and d(description="Navigate up").exists() and d(resourceId="de.rampro.activitydiary:id/activity_name").exists()
    )
    @rule()
    def delete_activity(self):
        activity_count = int(d(resourceId="de.rampro.activitydiary:id/activity_name").count)
        print("activity count: " + str(activity_count))
        selected_activity_index = random.randint(0, activity_count - 1)
        selected_activity = d(resourceId="de.rampro.activitydiary:id/activity_name")[selected_activity_index]
        selected_activity_name = selected_activity.get_text()
        print("selected activity: " + selected_activity_name)
        selected_activity.click()
        
        d(resourceId="de.rampro.activitydiary:id/action_edit_delete").click()
        
        assert d(text=selected_activity_name).exists() == False, "activity not deleted"

if __name__ == "__main__":
    t = Test()
    
    setting = Setting(
        apk_path="./apk/activitydiary/1.4.0.apk",
        device_serial="emulator-5554",
        output_dir="output/activitydiary/59/1",
        policy_name="random"
    )
    start_kea(t,setting)
    
