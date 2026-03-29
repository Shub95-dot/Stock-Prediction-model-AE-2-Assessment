import traceback
import test_inference
try:
    test_inference.run_test()
except Exception:
    with open("error_log.txt", "w") as f:
        traceback.print_exc(file=f)
    print("Error caught and written to error_log.txt")
