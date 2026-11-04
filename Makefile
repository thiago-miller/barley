CC             = gcc
CFLAGS         = -Wall -Wextra -O3 -fPIC -I$(INC_DIR)
LDFLAGS        = -lOpenCL
LDLIBS_TEST    = -lcheck -lOpenCL

SRC_DIR        = src
INC_DIR        = include
KERNELS_DIR    = kernels
TEST_DIR       = tests
BUILD_DIR      = build
CP_KERNELS_DIR = $(BUILD_DIR)/kernels

SRCS           = $(wildcard $(SRC_DIR)/*.c)
OBJS           = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

KERNELS        = $(wildcard $(KERNELS_DIR)/*.cl)
CP_KERNELS     = $(KERNELS:$(KERNELS_DIR)/%.cl=$(CP_KERNELS_DIR)/%.cl)

TEST_SRCS      = $(wildcard $(TEST_DIR)/*.c)
TEST_OBJS      = $(TEST_SRCS:$(TEST_DIR)/%.c=$(BUILD_DIR)/%.o)
TEST_TARGET    = $(BUILD_DIR)/check_bcl

LIB_SHARED     = $(BUILD_DIR)/libbcl.so
LIB_STATIC     = $(BUILD_DIR)/libbcl.a

.PHONY: all clean test

all: $(LIB_SHARED) $(LIB_STATIC) $(CP_KERNELS)

$(LIB_SHARED): $(OBJS)
	$(CC) -shared $(OBJS) $(LDFLAGS) -o $@

$(LIB_STATIC): $(OBJS)
	ar rcs $@ $(OBJS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(CP_KERNELS_DIR)/%.cl: $(KERNELS_DIR)/%.cl | $(CP_KERNELS_DIR)
	cp $< $@

$(BUILD_DIR) $(CP_KERNELS_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)

test: $(TEST_TARGET)
	@echo "Running tests..."
	@./$^ && echo "Tests OK" \
		|| exit 1

$(TEST_TARGET): $(TEST_OBJS) $(LIB_STATIC)
	$(CC) $(CFLAGS) $^ $(LDLIBS_TEST) -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@
