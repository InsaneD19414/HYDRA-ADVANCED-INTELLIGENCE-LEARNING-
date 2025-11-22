import Foundation

class HydraAgent: ObservableObject {
    func runAll(portfolio: Portfolio) async {
        await cryptoAgent(portfolio: portfolio)
        await goldAgent(portfolio: portfolio)
    }
    private func cryptoAgent(portfolio: Portfolio) async {
        let profit = Double.random(in: 50...200)
        portfolio.bitcoin += profit / 50000
        portfolio.trades.append(Trade(asset: "Bitcoin", profit: profit))
        portfolio.totalWealth += profit
    }
    private func goldAgent(portfolio: Portfolio) async {
        let profit = Double.random(in: 30...150)
        portfolio.goldBars += Int(profit / 50)
        portfolio.trades.append(Trade(asset: "Gold", profit: profit))
        portfolio.totalWealth += profit
    }
}