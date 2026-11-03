#include <bcl/utils.h>

#include <stdlib.h>

char *
bcl_file_slurp (const char *filename, size_t *out_size, BclError *out_err)
{
	BclError err = BCL_SUCCESS;
	FILE *fp = NULL;
	char *buf = NULL;
	size_t read_bytes = 0;
	size_t size = 0;
	long bytes = 0;

	fp = fopen (filename, "r");
	if (fp == NULL)
		{
			err = errno;
			goto ERR;
		}

	if (fseek (fp, 0L, SEEK_END) == EOF)
		{
			err = errno;
			goto ERR;
		}

	bytes = ftell (fp);
	if (bytes == EOF)
		{
			err = errno;
			goto ERR;
		}

	size = (size_t) bytes;

	buf = malloc (size + 1);
	if (buf == NULL)
		{
			err = errno;
			goto ERR;
		}

	if (fseek (fp, 0L, SEEK_SET) == EOF)
		{
			err = errno;
			goto ERR;
		}

	read_bytes = fread (buf, sizeof (char), size, fp);
	if (read_bytes != size)
		{
			err = EIO;
			goto ERR;
		}

	buf[size] = '\0';
	fclose (fp);

RET:
	if (out_size != NULL)
		*out_size = size;

	if (out_err != NULL)
		*out_err = err;

	return buf;

ERR:
	if (fp != NULL)
		fclose (fp);

	if (buf != NULL)
		free (buf);

	goto RET;
}
