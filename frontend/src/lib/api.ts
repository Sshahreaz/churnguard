export interface DashboardSummary {
  total_customers: number;
  high_risk_count: number;
  medium_risk_count: number;
  low_risk_count: number;
  overall_churn_rate: number;
  average_churn_probability: number;
}

export type RiskLevel = "High" | "Medium" | "Low";

export interface CustomerRisk {
  customer_id: number;
  churn_probability: number;
  risk_level: RiskLevel;
  top_3_reasons: string[];
}

const API_BASE_URL = "http://localhost:8000";

export async function getDashboardSummary(): Promise<DashboardSummary> {
  const res = await fetch(`${API_BASE_URL}/dashboard/summary`, {
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch dashboard summary: ${res.status}`);
  }

  return res.json() as Promise<DashboardSummary>;
}

export async function getCustomerRisk(
  riskLevel?: RiskLevel,
): Promise<CustomerRisk[]> {
  const url = new URL(`${API_BASE_URL}/customers/risk`);
  if (riskLevel) {
    url.searchParams.set("risk_level", riskLevel);
  }

  const res = await fetch(url.toString(), {
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch customer risk: ${res.status}`);
  }

  return res.json() as Promise<CustomerRisk[]>;
}
