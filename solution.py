def generate_repayment_schedule(principal, annual_rate, months):
    # Sửa lỗi: Lãi suất hàng tháng và số tiền trả cố định hàng tháng chưa được tính toán chính xác
    # Giải pháp: Sử dụng công thức tính EMI và lãi suất hàng tháng theo đúng quy tắc
    
    monthly_rate = annual_rate / 12 / 100
    emi = principal * (monthly_rate * ((1 + monthly_rate) ** months)) / (((1 + monthly_rate) ** months) - 1)
    emi = round(emi, 2)
    
    schedule = []
    remaining_balance = principal
    
    for month in range(1, months + 1):
        interest_paid = remaining_balance * monthly_rate
        principal_paid = emi - interest_paid
        remaining_balance -= principal_paid
        
        # Làm tròn số tiền gốc và lãi phải trả ở mỗi kỳ
        schedule.append({
            "month": month,
            "principal_paid": round(principal_paid, 2),
            "interest_paid": round(interest_paid, 2),
            "remaining_balance": round(remaining_balance, 2)
        })
    
    # Kiểm tra và điều chỉnh số dư nợ cuối cùng trong kỳ cuối
    if remaining_balance < principal_paid:
        last_principal_paid = remaining_balance
        last_interest_paid = (remaining_balance + principal_paid) * monthly_rate - remaining_balance
        schedule[-1].update({
            "principal_paid": round(last_principal_paid, 2),
            "interest_paid": round(last_interest_paid, 2),
            "remaining_balance": 0.00
        })
    
    return {"emi": emi, "schedule": schedule}