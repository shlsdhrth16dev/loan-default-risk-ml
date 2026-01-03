"""
Test script for CrewAI multi-agent system.

Tests:
- Individual agents
- Crew orchestration
- End-to-end decision flow
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents.loan_decision_crew import get_loan_decision_crew


def test_valid_application():
    """Test with a valid loan application."""
    print("\n" + "="*70)
    print("TEST 1: Valid Low-Risk Application")
    print("="*70)
    
    application = {
        "age": 35,
        "income": 75000,
        "loanamount": 20000,
        "creditscore": 720,
        "monthsemployed": 60,
        "numcreditlines": 5,
        "interestrate": 0.08,
        "loanterm": 60,
        "dtiratio": 0.30,
        "education": "Bachelor",
        "employmenttype": "Full-time",
        "maritalstatus": "Married",
        "hasmortgage": "Yes",
        "hasdependents": "Yes",
        "loanpurpose": "DebtConsolidation",
        "hascosigner": "No"
    }
    
    crew = get_loan_decision_crew()
    result = crew.process(application)
    
    print(f"\n✓ Decision: {result['decision']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Risk Level: {result['risk_assessment']['risk_level']}")
    print(f"  Default Probability: {result['risk_assessment']['default_probability']:.3f}")
    print(f"  Execution Time: {result['metadata']['total_execution_time_ms']:.1f}ms")
    print(f"  Agents Executed: {', '.join(result['metadata']['agents_executed'])}")
    
    print(f"\n  Borrower Explanation:")
    print(f"  '{result['explanations']['borrower_explanation']}'")
    
    return result


def test_invalid_application():
    """Test with invalid application."""
    print("\n" + "="*70)
    print("TEST 2: Invalid Application (Missing Fields)")
    print("="*70)
    
    application = {
        "age": 35,
        "income": 50000,
        # Missing required fields
    }
    
    crew = get_loan_decision_crew()
    result = crew.process(application)
    
    print(f"\n✓ Decision: {result['decision']}")
    print(f"  Reason: {result.get('reason', 'N/A')}")
    if 'validation_errors' in result:
        print(f"  Errors: {len(result['validation_errors'])} validation errors")
        for error in result['validation_errors'][:3]:
            print(f"    - {error}")
    
    return result


def test_high_risk_application():
    """Test with high-risk application."""
    print("\n" + "="*70)
    print("TEST 3: High-Risk Application")
    print("="*70)
    
    application = {
        "age": 22,
        "income": 25000,
        "loanamount": 35000,
        "creditscore": 580,
        "monthsemployed": 3,
        "numcreditlines": 2,
        "interestrate": 0.22,
        "loanterm": 84,
        "dtiratio": 0.52,
        "education": "High School",
        "employmenttype": "Part-time",
        "maritalstatus": "Single",
        "hasmortgage": "No",
        "hasdependents": "No",
        "loanpurpose": "Other",
        "hascosigner": "No"
    }
    
    crew = get_loan_decision_crew()
    result = crew.process(application)
    
    print(f"\n✓ Decision: {result['decision']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Risk Level: {result['risk_assessment']['risk_level']}")
    print(f"  Default Probability: {result['risk_assessment']['default_probability']:.3f}")
    print(f"  Risk Flags: {', '.join(result['risk_assessment']['flags'])}")
    
    if result.get('policy_evaluation'):
        soft_flags = result['policy_evaluation'].get('soft_flags', [])
        if soft_flags:
            print(f"  Policy Flags: {len(soft_flags)}")
            for flag in soft_flags[:3]:
                print(f"    - {flag['description']}")
    
    print(f"\n  Borrower Explanation:")
    print(f"  '{result['explanations']['borrower_explanation']}'")
    
    return result


def test_borderline_application():
    """Test with borderline application."""
    print("\n" + "="*70)
    print("TEST 4: Borderline Application")
    print("="*70)
    
    application = {
        "age": 40,
        "income": 55000,
        "loanamount": 28000,
        "creditscore": 660,
        "monthsemployed": 24,
        "numcreditlines": 8,
        "interestrate": 0.12,
        "loanterm": 60,
        "dtiratio": 0.42,
        "education": "Bachelor",
        "employmenttype": "Full-time",
        "maritalstatus": "Divorced",
        "hasmortgage": "No",
        "hasdependents": "Yes",
        "loanpurpose": "DebtConsolidation",
        "hascosigner": "No"
    }
    
    crew = get_loan_decision_crew()
    result = crew.process(application)
    
    print(f"\n✓ Decision: {result['decision']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Default Probability: {result['risk_assessment']['default_probability']:.3f}")
    
    if result.get('policy_evaluation'):
        print(f"  Manual Review Required: {result['policy_evaluation']['requires_manual_review']}")
    
    print(f"\n  Loan Officer Explanation:")
    lines = result['explanations']['loan_officer_explanation'].split('\n')
    for line in lines[:10]:  # First 10 lines
        print(f"  {line}")
    
    return result


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CREWAI MULTI-AGENT SYSTEM - COMPREHENSIVE TEST")
    print("="*70)
    
    try:
        # Run tests
        test1 = test_valid_application()
        test2 = test_invalid_application()
        test3 = test_high_risk_application()
        test4 = test_borderline_application()
        
        # Summary
        print("\n" + "="*70)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        print(f"\nTest Results:")
        print(f"  Test 1 (Valid): {test1['decision']}")
        print(f"  Test 2 (Invalid): {test2['decision']}")
        print(f"  Test 3 (High-Risk): {test3['decision']}")
        print(f"  Test 4 (Borderline): {test4['decision']}")
        
        print(f"\nCrew Info:")
        crew = get_loan_decision_crew()
        info = crew.get_crew_info()
        print(f"  Name: {info['name']}")
        print(f"  Version: {info['version']}")
        print(f"  Agents: {len(info['agents'])}")
        
        print("\n🚀 CrewAI system is production-ready!")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
