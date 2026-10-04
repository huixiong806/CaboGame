// Effects are driven by new, already-filtered public log entries from the SSE board.
// Keep the cursor outside the swapped fragment so reconnects do not replay actions.
(() => {
  const board = document.getElementById('board');
  const notices = document.getElementById('action-notices');
  const turnStatus = document.getElementById('turn-status');
  if (!board || !notices) return;
  let lastLogId = 0;
  let lastMode = '';
  let selectedTarget = '';
  let localAction = false;
  const effects = new Map();
  const icons = { info: '🂠', peek: '👁', power: '👁', swap: '⇄', cabo: '📣', score: '🏁' };
  const numbers = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩'];

  function actionCue(line) {
    const text = line.textContent.trim();
    const kind = line.dataset.actionKind;
    // Metadata comes only from the viewer-filtered public log, never private ranks.
    const keys = line.dataset.actionTargets.split(' ').filter(key => /^(p\d+|c\d+-\d+|deck|discard)$/.test(key));
    const actorKey = keys.find(key => /^p\d+$/.test(key));
    const actorSeat = actorKey && board.querySelector(`[data-effect-key="${actorKey}"]`);
    const actor = actorSeat?.dataset.playerName || '';
    const groups = new Map();
    const cards = keys.filter(key => key.startsWith('c'));
    for (const key of cards) {
      const [player, slot] = key.slice(1).split('-');
      const group = groups.get(player) || [];
      group.push(numbers[Number(slot)] || `第${Number(slot) + 1}张`);
      groups.set(player, group);
    }
    const targets = [...groups].map(([player, slots]) => {
      const seat = board.querySelector(`[data-effect-key="p${player}"]`);
      const name = `p${player}` === actorKey ? '' : (seat?.dataset.playerName || `P${Number(player) + 1}`);
      return `${name ? `${name} · ` : ''}${slots.join('')}`;
    });
    let verb = '', result = '', icon = icons[kind] || '•';
    if (kind === 'peek') verb = '看牌';
    else if (kind === 'power') verb = text.includes('【间谍】') ? '间谍' : '偷看';
    else if (kind === 'swap') {
      if (text.includes('发动【交换】')) verb = '换牌';
      else if (text.includes('交换失败')) { verb = '合并失败'; result = '手牌 +1'; }
      else if (text.includes('点数相同')) { verb = '合并成功'; result = '手牌减少'; }
      else {
        verb = '替换';
        const exposed = text.match(/亮出 (\d+)$/) || text.match(/的 (\d+)，换入/);
        if (exposed) result = `换出 ${exposed[1]}`;
      }
    } else if (kind === 'cabo') verb = 'CABO';
    else if (kind === 'score') verb = '本轮结算';
    else if (keys.includes('deck')) verb = '摸牌';
    else if (keys.includes('discard')) { verb = '弃牌'; result = text.match(/弃置了 (\d+)/)?.[1] || ''; }
    // Regular draw/discard/turn changes need no marker anywhere on the table.
    const marked = ['peek', 'power', 'swap'].includes(kind) ? cards.slice(0, 2) : [];
    return {text, kind, actor, verb, result, targets, marked, icon, keys};
  }

  function showNotice(cue) {
    const item = document.createElement('div');
    item.className = `action-notice notice-${cue.kind}`;
    item.title = cue.text;
    const icon = document.createElement('span');
    icon.className = 'notice-icon';
    icon.textContent = cue.icon;
    icon.setAttribute('aria-hidden', 'true');
    item.append(icon);
    for (const [className, text] of [
      ['notice-actor', cue.actor], ['notice-verb', cue.verb],
      ...cue.targets.flatMap((target, index) => index && cue.verb === '换牌'
        ? [['notice-verb', '⇄'], ['notice-target', target]] : [['notice-target', target]]),
      ['notice-result', cue.result],
    ]) {
      if (!text) continue;
      const part = document.createElement('span');
      part.className = className;
      part.textContent = text;
      item.append(part);
    }
    notices.replaceChildren(item);
  }

  function markTargets(cue) {
    // Replace rather than accumulate: never frame the acting player or next player.
    effects.clear();
    board.querySelectorAll('.action-highlight').forEach(el => {
      el.classList.remove('action-highlight');
      delete el.dataset.effectIcon;
    });
    for (const key of cue.marked) effects.set(key, {icon: cue.icon, until: Date.now() + 1600});
  }

  function chooseTarget(player) {
    selectedTarget = player;
    board.querySelectorAll('[data-target-tab]').forEach(tab => {
      const selected = tab.dataset.targetTab === player;
      tab.classList.toggle('active', selected);
      tab.setAttribute('aria-pressed', String(selected));
    });
    board.querySelectorAll('[data-target-hand]').forEach(hand => {
      hand.hidden = hand.dataset.targetHand !== player;
    });
  }

  function updateWorkArea(initial) {
    const game = board.querySelector('.game');
    const mode = game?.dataset.interactionMode || 'lobby';
    const turn = board.querySelector('.seat.turn');
    const ownTurn = turn?.classList.contains('me');
    const prompts = {
      initial: '开局查看 · 点选自己的 2 张牌',
      idle: '轮到你了',
      drawn: '你的回合 · 处理摸牌',
      peek: '偷看 · 选自己的牌',
      own: '换牌 · 先选自己的牌',
      other: '选择目标牌',
      multi: '选牌后确认交换',
      confirm: '确认 Cabo',
      'round-end': '本轮结束 · 查看结算',
      'game-over': '大局结束 · 查看胜负',
      lobby: '等待开局',
    };
    if (turnStatus) {
      const status = prompts[mode] || (turn ? `${turn.dataset.playerName} 正在行动` : '等待其他玩家完成开局查看');
      if (turnStatus.textContent !== status) turnStatus.textContent = status;
      turnStatus.classList.toggle('your-turn', Boolean(ownTurn) || mode === 'initial');
    }
    const tabs = [...board.querySelectorAll('[data-target-tab]')];
    if (tabs.length) chooseTarget(tabs.some(tab => tab.dataset.targetTab === selectedTarget) ? selectedTarget : tabs[0].dataset.targetTab);
    // Scroll only after a local operation enters a different step, never on AI updates
    // or every selected-card toggle. This keeps the hand and nearby controls in reach.
    if (!initial && localAction && mode !== lastMode && window.matchMedia('(max-width: 520px)').matches) {
      const focus = mode === 'other' ? board.querySelector('.target-picker') : board.querySelector('.action-panel');
      if (focus && ['initial', 'drawn', 'multi', 'own', 'other', 'peek', 'confirm'].includes(mode)) {
        const bottom = focus.getBoundingClientRect().bottom;
        if (bottom > window.innerHeight - 12) focus.scrollIntoView({block: 'end', behavior: 'instant'});
      }
    }
    lastMode = mode;
    localAction = false;
  }

  function paintEffects() {
    const now = Date.now();
    for (const [key, effect] of effects) {
      if (effect.until <= now) {
        effects.delete(key);
        board.querySelectorAll(`[data-effect-key="${key}"]`).forEach(el => {
          el.classList.remove('action-highlight');
          delete el.dataset.effectIcon;
        });
        continue;
      }
      board.querySelectorAll(`[data-effect-key="${key}"]`).forEach(el => {
        el.classList.add('action-highlight');
        el.dataset.effectIcon = effect.icon;
      });
    }
  }

  function inspectBoard(initial = false) {
    const lines = [...board.querySelectorAll('[data-log-id]')];
    const newest = Math.max(lastLogId, ...lines.map(el => Number(el.dataset.logId)));
    const fresh = [];
    if (!initial) {
      for (const line of lines.reverse()) {
        if (Number(line.dataset.logId) <= lastLogId) continue;
        // Private peek results remain in their existing private panel/log only.
        const cue = actionCue(line);
        if ((!cue.keys.length && cue.kind !== 'score') || !cue.verb) continue;
        fresh.push(cue);
      }
      if (fresh.length > 1 && fresh.every(e => e.kind === 'peek')) {
        showNotice({text: '开局查看完成，具体牌位见牌局动态', kind: 'peek', icon: '👁', actor: '', verb: '开局查看完成', targets: [], result: ''});
        markTargets({marked: []});
      } else if (fresh.length) {
        const latest = fresh[fresh.length - 1];
        showNotice(latest);
        markTargets(latest);
      }
    } else {
      const latest = lines.find(line => line.dataset.actionTargets || line.dataset.actionKind === 'score');
      if (latest) {
        showNotice(actionCue(latest));
      } else notices.textContent = '等待开局';
    }
    lastLogId = newest;
    updateWorkArea(initial);
    paintEffects();
  }

  board.addEventListener('click', event => {
    const tab = event.target.closest('[data-target-tab]');
    if (tab) chooseTarget(tab.dataset.targetTab);
  });
  board.addEventListener('submit', () => { localAction = true; });

  inspectBoard(true);
  // SSE can coalesce updates. Inspect the actual swapped DOM, including all new log rows.
  const observer = new MutationObserver(records => {
    if (records.some(record => record.addedNodes.length || record.removedNodes.length)) inspectBoard();
  });
  observer.observe(board, { childList: true, subtree: true });
  window.setInterval(paintEffects, 250);
})();
