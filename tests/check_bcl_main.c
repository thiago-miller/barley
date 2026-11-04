#include <check.h>
#include "check_bcl.h"

int
main (void)
{
	int number_failed = 0;
	SRunner *sr = NULL;

	// Create runner
	sr = srunner_create (make_tensor_ops_suite ());
	/*srunner_add_suite (sr, make_aa_suite ());*/
	srunner_set_tap (sr, "-");

	srunner_run_all (sr, CK_ENV);
	number_failed = srunner_ntests_failed (sr);
	srunner_free (sr);

	return number_failed;
}
