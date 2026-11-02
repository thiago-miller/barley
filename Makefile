CC         = gcc
CFLAGS     = -Wall -Wextra -O3 -fPIC -I$(INC_DIR)
LDFLAGS    = -lOpenCL

SRC_DIR    = src
INC_DIR    = include
BUILD_DIR  = build

SRCS       = $(wildcard $(SRC_DIR)/*.c)
OBJS       = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

LIB_SHARED = $(BUILD_DIR)/libbcl.so
LIB_STATIC = $(BUILD_DIR)/libbcl.a

.PHONY: all clean

all: $(LIB_SHARED) $(LIB_STATIC)

$(LIB_SHARED): $(OBJS)
	$(CC) -shared $(OBJS) $(LDFLAGS) -o $@

$(LIB_STATIC): $(OBJS)
	ar rcs $@ $(OBJS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
