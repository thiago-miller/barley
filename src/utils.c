#include <bcl/utils.h>

#include <stdlib.h>
#include <errno.h>

char *
bcl_file_slurp (const char *filename, size_t *out_size, BclError *out_err)
{
	BclError err = BCL_SUCCESS;
	FILE *fp = NULL;
	char *buf = NULL;
	size_t read_bytes = 0;
	size_t size = 0;

	fp = fopen (filename, "r");
	if (fp == NULL)
		{
			err = errno;
			goto RET;
		}

	if (fseek (fp, 0L, SEEK_END) == EOF)
		{
			err = errno;
			goto RET;
		}

	size = ftell (fp);
	if (size == EOF)
		{
			err = errno;
			goto RET;
		}

	buf = malloc (size + 1);
	if (buf == NULL)
		{
			err = errno;
			goto RET;
		}

	if (fseek (fp, 0L, SEEK_SET) == EOF)
		{
			err = errno;
			goto RET;
		}

	read_bytes = fread (buf, sizeof (char), size, fp);
	if (read_bytes != size)
		{
			err = EIO;
			goto RET;
		}

	buf[size] = '\0';

RET:
	if (fp != NULL)
		fclose (fp);

	if (out_size != NULL)
		*out_size = size;

	if (out_err != NULL)
		*out_err = err;

	return buf;
}
