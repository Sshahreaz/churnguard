import { getDashboardSummary } from "@/lib/api";
import RiskTable from "@/components/RiskTable";

export default async function Home() {
  const summary = await getDashboardSummary();

  const cards = [
    {
      label: "Total Customers",
      value: summary.total_customers.toLocaleString(),
      accent: "border-slate-200",
      valueClass: "text-slate-900",
    },
    {
      label: "High Risk",
      value: summary.high_risk_count.toLocaleString(),
      accent: "border-l-4 border-l-red-500 border-slate-200",
      valueClass: "text-red-700",
    },
    {
      label: "Medium Risk",
      value: summary.medium_risk_count.toLocaleString(),
      accent: "border-l-4 border-l-amber-500 border-slate-200",
      valueClass: "text-amber-700",
    },
    {
      label: "Churn Rate",
      value: `${summary.overall_churn_rate.toFixed(2)}%`,
      accent: "border-slate-200",
      valueClass: "text-slate-900",
    },
  ];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <main className="mx-auto max-w-6xl px-4 py-10 sm:px-6 lg:px-8">
        <header className="mb-8">
          <h1 className="text-3xl font-semibold tracking-tight text-slate-900">
            ChurnGuard
          </h1>
          <p className="mt-1 text-base text-slate-600">
            Customer Retention Dashboard
          </p>
        </header>

        <section
          aria-label="Key metrics"
          className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4"
        >
          {cards.map((card) => (
            <article
              key={card.label}
              className={`rounded-lg border bg-white p-5 shadow-sm ${card.accent}`}
            >
              <p className="text-sm font-medium text-slate-500">{card.label}</p>
              <p className={`mt-2 text-3xl font-semibold tabular-nums ${card.valueClass}`}>
                {card.value}
              </p>
            </article>
          ))}
        </section>

        <section className="mt-10" aria-labelledby="risk-list-heading">
          <h2
            id="risk-list-heading"
            className="mb-4 text-lg font-semibold text-slate-900"
          >
            Customer Risk List
          </h2>
          <RiskTable />
        </section>
      </main>
    </div>
  );
}
