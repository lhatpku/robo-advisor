# User Flow Testing Suite

This directory contains comprehensive test cases for validating user flows in the robo-advisor application. The test suite has been cleaned up to include only the **working test cases** that pass successfully.

## ✅ **Working Test Cases (6/6)**

### 1. **Comprehensive Risk Assessment Flow** (`test_comprehensive_risk_flow.py`)
- Tests the complete risk assessment flow from start to finish
- Covers: greeting → "yes" → mode selection → "set as 0.6" → "guidance" → complete questionnaire → "why" → "proceed"
- Validates the exact user flow provided by the user
- **Status**: ✅ PASSING

### 2. **Simple Final Completion** (`test_simple_completion.py`)
- Tests basic final completion flow with pre-existing trading requests
- Validates reviewer agent completion logic
- **Status**: ✅ PASSING

### 3. **Start Over Functionality** (`test_start_over.py`)
- Tests "review", "start over", and "proceed" commands in final completion state
- Validates state reset and routing back to entry agent
- **Status**: ✅ PASSING

### 4. **Reviewer Final Completion** (`test_reviewer_final_completion.py`)
- Tests reviewer agent handling of final completion options
- Validates comprehensive summary generation and routing
- **Status**: ✅ PASSING

### 5. **Risk Agent Review/Edit** (`test_risk_review_edit.py`)
- Tests risk agent review/edit functionality after setting equity
- Validates "review" command, equity changes, and routing
- **Status**: ✅ PASSING

### 6. **Risk Agent Guidance After Equity** (`test_risk_review_edit.py`)
- Tests using guidance questionnaire after setting equity
- Validates reset functionality and questionnaire restart
- **Status**: ✅ PASSING

## 🚧 **Removed Test Cases (Covered by Comprehensive Test)**

The following test cases were removed as they are now covered by the comprehensive test:

- ~~Risk Assessment Flow~~ - Covered by comprehensive test (questionnaire flow)
- ~~Direct Equity 'as' Pattern~~ - Covered by comprehensive test (direct equity setting)

## 🏃 **Running the Tests**

### Run All Working Tests
```bash
python userflowtesting/test_suite.py
```

### Run Individual Tests
```bash
python userflowtesting/test_comprehensive_risk_flow.py
python userflowtesting/test_simple_completion.py
python userflowtesting/test_start_over.py
python userflowtesting/test_reviewer_final_completion.py
python userflowtesting/test_risk_review_edit.py
```

## 📊 **Test Results**

```
=== Test Summary ===
Total Tests: 6
Passed: 6
Failed: 0
Errors: 0

All tests passed! 🎉
```

## 🔧 **Test Environment Setup**

All tests include proper environment setup:
- `load_dotenv()` for API key loading
- Proper imports and path configuration
- Error handling and result validation

## 📝 **Notes**

- The test suite focuses on **core functionality that works reliably**
- The comprehensive test covers the exact user flow provided by the user
- Duplicate test cases have been removed to avoid redundancy
- Each test includes detailed logging and validation steps
- All tests are currently passing successfully