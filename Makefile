CC             = gcc
CFLAGS         = -Wall -Wextra -O3 -fPIC -I$(INC_DIR)
LDFLAGS        = -lOpenCL

SRC_DIR        = src
INC_DIR        = include
KERNELS_DIR    = kernels
BUILD_DIR      = build
CP_KERNELS_DIR = $(BUILD_DIR)/kernels

SRCS           = $(wildcard $(SRC_DIR)/*.c)
OBJS           = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

KERNELS        = $(wildcard $(KERNELS_DIR)/*.cl)
CP_KERNELS     = $(KERNELS:$(KERNELS_DIR)/%.cl=$(CP_KERNELS_DIR)/%.cl)

LIB_SHARED     = $(BUILD_DIR)/libbcl.so
LIB_STATIC     = $(BUILD_DIR)/libbcl.a

.PHONY: all clean

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
