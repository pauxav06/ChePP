EXE ?= ChePP
TARGET := ChePP
BUILD_DIR := build

BIN_SINGLE := $(BUILD_DIR)/engine/src/$(TARGET)
BIN_MULTI  := $(BUILD_DIR)/engine/src/Release/$(TARGET)

CMAKE_ARGS := -DCMAKE_BUILD_TYPE=Release

ifdef CC
CMAKE_ARGS += -DCMAKE_C_COMPILER=$(CC)
endif

ifdef CXX
CMAKE_ARGS += -DCMAKE_CXX_COMPILER=$(CXX)
endif

all:
	cmake -B $(BUILD_DIR) -S . $(CMAKE_ARGS) -DSTATIC -DARCH=native
	cmake --build $(BUILD_DIR) --target $(TARGET) --config Release
	@if [ -f $(BIN_SINGLE) ]; then \
		cp $(BIN_SINGLE) ./$(EXE); \
	elif [ -f $(BIN_MULTI) ]; then \
		cp $(BIN_MULTI) ./$(EXE); \
	else \
		echo "Error: executable not found"; \
		exit 1; \
	fi

clean:
	rm -rf $(BUILD_DIR) $(EXE)